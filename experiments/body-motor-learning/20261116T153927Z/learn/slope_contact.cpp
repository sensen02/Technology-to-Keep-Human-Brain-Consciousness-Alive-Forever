#include <algorithm>
#include <cmath>
#include <mujoco/mujoco.h>

extern "C" void slope_contacts(const mjModel* model, const mjData* data,
                               int floor, const int* feet, const double* velocity,
                               double* result) {
    std::fill(result, result + 13, 0.);
    const mjtNum* rotation = data->geom_xmat + 9 * floor;
    const double normal[3] = {rotation[2], rotation[5], rotation[8]};
    double jac[3 * 48];
    double wrench[6];
    for (int i = 0; i < data->ncon; ++i) {
        const mjContact& contact = data->contact[i];
        const int a = contact.geom[0], b = contact.geom[1];
        if (contact.efc_address < 0 || a < 0 || b < 0 || (a == floor) == (b == floor)) continue;
        const int geom = a == floor ? b : a;
        mj_contactForce(model, data, i, wrench);
        const double sign = a == floor ? 1. : -1.;
        double support = 0.;
        for (int axis = 0; axis < 3; ++axis) {
            const double component = contact.frame[axis] * wrench[0] +
                contact.frame[3 + axis] * wrench[1] + contact.frame[6 + axis] * wrench[2];
            support += normal[axis] * component * sign;
        }
        support = std::max(0., support);
        const int leg = feet[geom];
        if (leg < 0) {
            result[12] += support;
            continue;
        }
        mj_jac(model, data, jac, nullptr, contact.pos, model->geom_bodyid[geom]);
        double point_velocity[3] = {};
        for (int axis = 0; axis < 3; ++axis) {
            for (int dof = 0; dof < model->nv; ++dof)
                point_velocity[axis] += jac[axis*model->nv + dof] * velocity[dof];
        }
        const double projected = mju_dot3(point_velocity, normal);
        for (int axis = 0; axis < 3; ++axis) point_velocity[axis] -= normal[axis] * projected;
        result[leg] += support;
        result[leg + 6] += support * mju_norm3(point_velocity);
    }
}
