#include <algorithm>
#include <cmath>
#include <mujoco/mujoco.h>

// Position/adhesion references use the model's native actuator interface.
extern "C" int simulate(const mjModel* m, mjData* d, const double* start,
                        const double* actions, int count, int native_steps,
                        int thorax, const int* legs, double* result, double* end,
                        double* trajectory, int record_stride) {
    const int nu = m->nu;
    constexpr unsigned int state = mjSTATE_INTEGRATION;
    mj_setState(m,d,start,state);
    mj_forward(m,d);
    double best = -10.;
    for (int i = 0; i < count; ++i) {
        const double* command = actions+nu*i;
        std::copy(command,command+nu,d->ctrl);
        for (int k = 0; k < native_steps; ++k) {
            mj_step(m,d);
            if (trajectory != nullptr && (i*native_steps+k+1) % record_stride == 0) {
                int ix = (i*native_steps+k+1) / record_stride - 1;
                std::copy(d->qpos,d->qpos+m->nq,trajectory + ix * m->nq);
            }
            best = std::max(best, d->xmat[9*thorax+8]);
        }
    }
    mj_forward(m,d);
    unsigned int supports = 0;
    int nonfoot = 0;
    for (int i=0; i<d->ncon; ++i) {
        const mjContact& c=d->contact[i];
        if (c.efc_address < 0) continue;
        const int a = c.geom[0], b = c.geom[1];
        if (a < 0 || b < 0 || (legs[a]==-2) == (legs[b]==-2)) continue;
        const int k = legs[a]==-2 ? legs[b] : legs[a];
        double f[6];
        mj_contactForce(m,d,i,f);
        if (f[0] < 1.e-9) continue;
        if (k < 0) ++nonfoot;
        else if (std::abs(c.frame[2])>.45) supports |= (1u << k);
    }
    result[0] = d->xmat[9*thorax+8];
    result[1] = d->subtree_com[3*thorax+2];
    result[2] = mju_norm3(d->qvel);
    result[3] = mju_norm3(d->qvel+3);
    result[4] = __builtin_popcount(supports);
    result[5] = nonfoot;
    result[6] = best;
    result[7] = d->time;
    mj_getState(m,d,end,state);
    for (const auto& w: d->warning) if (w.number) return 1;
    for (int i=0;i<m->nq;++i) if (!std::isfinite(d->qpos[i])) return 2;
    return 0;
}
