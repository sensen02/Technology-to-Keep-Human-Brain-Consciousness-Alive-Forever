#include <algorithm>
#include <cmath>
#include <mujoco/mujoco.h>

extern "C" int measure(const mjModel* m, mjData* d, int floor, int full,
                         const int* legs, const double* normal, double* output,
                         double* jacobian, int* grazes) {
    if (full) std::fill(output, output + 46, 0.);
    for (int i=0; i<m->nv; ++i) {
        if (d->qfrc_applied[i] != 0.) return -1;
    }
    for (int i=0; i<6*m->nbody; ++i) {
        if (d->xfrc_applied[i] != 0.) return -1;
    }
    for (const auto& warning : d->warning) {
        if (warning.number) return -2;
    }
    int nonfoot = 0;
    for (int i=0; i<d->ncon; ++i) {
        const mjContact& c = d->contact[i];
        if (c.efc_address < 0) continue;
        int other;
        double sign;
        if (c.geom[0] == floor) {
            other = c.geom[1];
            sign = 1.;
        } else if (c.geom[1] == floor) {
            other = c.geom[0];
            sign = -1.;
        } else {
            continue;
        }
        if (other < 0) continue;
        const int body = m->geom_bodyid[other];
        const int leg = legs[body];
        if (!full && leg >= 0) continue;
        double contact_force[6];
        mj_contactForce(m, d, i, contact_force);
        if (contact_force[0] <= 1.e-8) continue;
        if (leg < 0) {
            ++nonfoot;
            ++grazes[body];
            continue;
        }
        double force[3];
        mju_mulMatTVec(force, c.frame, contact_force, 3, 3);
        mju_scl3(force, force, sign);
        const double load = mju_dot3(force, normal);
        if (load <= 0.) continue;
        mj_jac(m, d, jacobian, nullptr, c.pos, body);
        double velocity[3], tangent[3];
        mju_mulMatVec(velocity, jacobian, d->qvel, 3, m->nv);
        mju_addScl3(tangent, velocity, normal, -mju_dot3(velocity, normal));
        output[leg] += load;
        output[6 + leg] += load * mju_norm3(tangent);
        mju_addTo3(output + 12 + 3*leg, force);
        mju_addToScl3(output + 30, c.frame, load*sign);
        mju_addToScl3(output + 33, c.pos, load);
    }
    return nonfoot;
}
