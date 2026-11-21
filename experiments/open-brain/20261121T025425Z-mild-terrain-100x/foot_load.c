#include <mujoco/mujoco.h>

void per_foot_normal_load(const mjModel *model, const mjData *data,
                          const int *ground, const int *leg, mjtNum *load) {
    for (int f = 0; f < 6; ++f) {
        load[f] = 0;
    }
    for (int i = 0; i < data->ncon; ++i) {
        const mjContact *c = data->contact + i;
        if (c->exclude) {
            continue;
        }
        const int g0 = c->geom[0];
        const int g1 = c->geom[1];
        if (g0 < 0 || g1 < 0) {
            continue;
        }
        const int foot = ground[g0] ? leg[g1] : (ground[g1] ? leg[g0] : -1);
        if (foot >= 0) {
            mjtNum wrench[6];
            mj_contactForce(model, data, i, wrench);
            load[foot] += wrench[0];
        }
    }
}
