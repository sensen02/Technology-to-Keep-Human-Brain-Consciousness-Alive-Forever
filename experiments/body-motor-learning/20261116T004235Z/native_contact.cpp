#include <algorithm>
#include <cmath>
#include <mujoco/mujoco.h>

// Uses the solve's contact frame and velocities from the beginning of the interval.
extern "C" void contact_totals(const mjModel* model, const mjData* data, const int* foot,
                               const double* velocity, double* result) {
    std::fill(result, result + 13, 0.);
    double jac[3 * 48];
    double load[6];
    for (int index = 0; index < data->ncon; ++index) {
        const auto& contact = data->contact[index];
        const int a = contact.geom[0], b = contact.geom[1];
        if (a < 0 || b < 0 || contact.efc_address < 0) continue;
        const bool a_ground = model->geom_type[a] == mjGEOM_PLANE;
        const bool b_ground = model->geom_type[b] == mjGEOM_PLANE;
        if (a_ground == b_ground) continue;
        mj_contactForce(model, data, index, load);
        const double z = (a_ground ? 1. : -1.) *
            (contact.frame[2] * load[0] + contact.frame[5] * load[1] + contact.frame[8] * load[2]);
        const double support = std::max(0., z);
        const int geom = a_ground ? b : a;
        const int leg = foot[geom];
        if (leg < 0) {
            result[12] += support;
            continue;
        }
        mj_jac(model, data, jac, nullptr, contact.pos, model->geom_bodyid[geom]);
        double vx = 0., vy = 0.;
        for (int j = 0; j < model->nv; ++j) {
            vx += jac[j] * velocity[j];
            vy += jac[model->nv + j] * velocity[j];
        }
        result[leg] += support;
        result[leg + 6] += support * std::hypot(vx,vy);
    }
}
