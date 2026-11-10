"""Independent fresh-process verification; no physical stepping or updates."""
from pathlib import Path
import json
from unittest.mock import patch
import torch
import continue_role_bc as c
import student_wave as sw
from identity import atomic_json, verify, sha256

def main():
    torch.set_num_threads(1)
    with patch.object(sw.Core,'tick',side_effect=AssertionError('physics forbidden')):
        c.preflight()
        parent,episodes=c.validate_parent(c.PARENT)
        result=json.loads((c.OUT/'RESULT.json').read_text())
        assert result['steps']==3000 and result['physical_steps']==0
        assert [r['step'] for r in result['checkpoints']]==[2250,2500,2750,3000]
        for row in result['checkpoints']:
            path=Path(row['path']); assert sha256(path)==row['sha256']
            payload=torch.load(path,map_location='cpu',weights_only=False)
            c.validate_aux_payload(payload,payload['extra']['aux_bc_config'],row['step'])
            assert payload['extra']['curve'][:2001]==parent['extra']['curve']
            assert c.exact(c.frozen_payload(payload),c.frozen_payload(parent))
        final=Path(result['checkpoint']); assert sha256(final)==result['checkpoint_sha256']
        payload=torch.load(final,map_location='cpu',weights_only=False)
        verify(payload['identity'])
        resumed=c.restore_validated(final,payload,episodes)
        assert c.exact(resumed['model'].state_dict(),payload['state']['model'])
        assert c.exact(resumed['head'].state_dict(),payload['extra']['aux_head_state_dict'])
        assert c.exact(resumed['optimizer'].state_dict(),payload['extra']['aux_bc_optimizer'])
        assert c.exact(c.rng_state(),payload['rng'])
        assert c.measure(resumed,3000)==result['final']
        assert c.optimizer_steps(resumed['optimizer'],resumed['model'],resumed['head'])==[3000]*13
        assert float(resumed['core'].d.time)==0.
        report=dict(all_passed=True,checkpoint=str(final),checkpoint_sha256=sha256(final),steps=3000,
                    all_four_checkpoints_validated=True,full_curve_prefix_0_to_2000_exact=True,
                    model_head_13Adam_RNG_metrics_exact=True,source_identity_verified=True,
                    physical_steps=0,updates=0,closed_loop='NOT_EVALUATED',final=result['final'])
        atomic_json(c.HERE/'SAVED_TRAINING_VERIFICATION.json',report)
        print(json.dumps(report),flush=True)

if __name__=='__main__': main()
