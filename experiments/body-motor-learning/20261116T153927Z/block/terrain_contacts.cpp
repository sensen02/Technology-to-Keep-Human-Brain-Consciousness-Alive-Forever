#include <algorithm>
#include <cmath>
#include <mujoco/mujoco.h>

// Columns [six Fz, six Fz*slip speed, non-pad Fz, six block top Fz,
// six block non-top normal forces, six named segment side forces,
// block force on other fly parts].
extern "C" void block_totals(const mjModel* model, const mjData* data, const int* pads,
                              const int* parts, const int* terrain, const double* velocity,
                              double* result) {
    std::fill(result, result + 32, 0.);
    double jac[3 * 48];
    double force[6];
    for (int index = 0; index < data->ncon; ++index) {
        const auto& c = data->contact[index];
        const int a = c.geom[0], b = c.geom[1];
        if (a < 0 || b < 0 || c.efc_address < 0) continue;
        const int role_a = terrain[a], role_b = terrain[b];
        if ((role_a > 0) == (role_b > 0)) continue;
        const int role = role_a > 0 ? role_a : role_b;
        const double sign = role_a > 0 ? 1. : -1.;
        const int geom = role_a > 0 ? b : a;
        mj_contactForce(model, data, index, force);
        const double z = sign *
            (c.frame[2] * force[0] + c.frame[5] * force[1] + c.frame[8] * force[2]);
        const double support = std::max(0., z);
        const int leg = pads[geom], part = parts[geom];
        if (role == 2) {
            const bool top = sign * c.frame[2] >= 0.85;
            if (leg >= 0) {
                result[(top ? 13 : 19) + leg] += top ? support : force[0];
            } else if (part >= 0 && !top) {
                result[25 + part] += force[0];
            } else {
                result[31] += force[0];
            }
        }
        if (leg < 0) {
            result[12] += support;
            continue;
        }
        mj_jac(model, data, jac, nullptr, c.pos, model->geom_bodyid[geom]);
        double vx = 0., vy = 0.;
        for (int j = 0; j < model->nv; ++j) {
            vx += jac[j] * velocity[j];
            vy += jac[model->nv+j] * velocity[j];
        }
        result[leg] += support;
        result[6 + leg] += support * std::hypot(vx,vy);
    }
}
