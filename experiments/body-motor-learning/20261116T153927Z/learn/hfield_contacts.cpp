#include <algorithm>
#include <cmath>
#include <mujoco/mujoco.h>

extern "C" void hfield_contacts(const mjModel* model, const mjData* data,
                                int field, const int* feet, const double* velocity,
                                double lo, double hi,
                                double* vertical, double* slope, double* flags) {
    std::fill(vertical, vertical + 13, 0.);
    std::fill(slope, slope + 13, 0.);
    std::fill(flags, flags + 19, 0.);
    double jac[3 * 48];
    double wrench[6];
    for (int i = 0; i < data->ncon; ++i) {
        const mjContact& contact = data->contact[i];
        const int a = contact.geom[0], b = contact.geom[1];
        if (contact.efc_address < 0 || a < 0 || b < 0 || (a == field) == (b == field)) continue;
        const int geom = a == field ? b : a;
        const double sign = a == field ? 1. : -1.;
        const double z_normal = sign * contact.frame[2];
        mj_contactForce(model, data, i, wrench);
        const double z = sign * (contact.frame[2]*wrench[0] + contact.frame[5]*wrench[1] + contact.frame[8]*wrench[2]);
        const double world = std::max(0., z);
        const bool incline = contact.pos[0] > lo && contact.pos[0] < hi;
        const int leg = feet[geom];
        if (leg < 0) {
            vertical[12] += world;
            if (incline) {
                slope[12] += wrench[0];
                flags[18] += wrench[0];
            }
            continue;
        }
        mj_jac(model, data, jac, nullptr, contact.pos, model->geom_bodyid[geom]);
        double speed[3] = {};
        for (int axis = 0; axis < (incline ? 3 : 2); ++axis) {
            for (int j = 0; j < model->nv; ++j) speed[axis] += jac[axis*model->nv + j]*velocity[j];
        }
        vertical[leg] += world;
        vertical[leg+6] += world*std::hypot(speed[0], speed[1]);
        if (!incline) continue;
        if (z_normal > .8) flags[leg] += world;
        else flags[leg+6] += wrench[0];
        const double component = mju_dot3(speed, contact.frame);
        for (int axis = 0; axis < 3; ++axis) speed[axis] -= component*contact.frame[axis];
        slope[leg] += wrench[0];
        slope[leg+6] += wrench[0]*mju_norm3(speed);
    }
}
