#include <algorithm>
#include <cmath>
#include <cstring>
#include <mujoco/mujoco.h>

// The frozen body's position-target integrator and per-substep joint PD.
extern "C" int drive(const mjModel* model, mjData* data, const double* goals, int ticks,
                     double* target, const int* qa, const int* va, const double* kp,
                     const double* kd, const double* limits, const double* bounds,
                     int mode, int th, const int* foot_geometry, double* trace) {
    const int n = model->nu;
    const int native_steps = static_cast<int>(std::lround(.001 / model->opt.timestep));
    double rate[42];
    for (int tick = 0; tick < ticks; ++tick) {
        if (tick % 5 == 0) {
            for (int joint = 0; joint < n; ++joint) {
                rate[joint] = std::clamp((goals[(tick / 5) * n + joint] - target[joint]) / .005,
                                         -2.0, 2.0);
            }
        }
        for (int joint = 0; joint < n; ++joint) {
            target[joint] = std::clamp(target[joint] + rate[joint] * .001,
                                       bounds[2 * joint], bounds[2 * joint + 1]);
        }
        for (int step = 0; step < native_steps; ++step) {
            for (int joint = 0; joint < n; ++joint) {
                const double raw = kp[joint] * (target[joint] - data->qpos[qa[joint]])
                                   - kd[joint] * data->qvel[va[joint]];
                data->ctrl[joint] = std::clamp(raw, -limits[joint], limits[joint]);
            }
            mj_step(model, data);
            if (mode == 0) mj_forward(model, data);
        }
        if ((tick + 1) % 5 != 0) continue;
        if (mode != 0 && (trace != nullptr || tick + 1 == ticks)) mj_forward(model, data);
        if (trace != nullptr) {
            double* row = trace + (tick / 5) * 10;
            row[0] = data->xmat[9 * th + 8];
            row[1] = data->subtree_com[3 * th + 2];
            row[2] = mju_norm3(data->qvel + 3);
            row[3] = mju_norm3(data->qvel);
            std::fill(row + 4, row + 10, 0.0);
            for (int c = 0; c < data->ncon; ++c) {
                const mjContact& contact = data->contact[c];
                const int a = contact.geom[0], b = contact.geom[1];
                if (a < 0 || b < 0 || contact.efc_address < 0) continue;
                const bool first = model->geom_type[a] == mjGEOM_PLANE;
                const bool second = model->geom_type[b] == mjGEOM_PLANE;
                if (first == second) continue;
                const int leg = foot_geometry[first ? b : a];
                if (leg < 0) continue;
                double local[6];
                mj_contactForce(model, data, c, local);
                double z = contact.frame[2] * local[0] + contact.frame[5] * local[1]
                    + contact.frame[8] * local[2];
                row[4 + leg] += std::max(0., first ? z : -z);
            }
        }
        if (!std::isfinite(data->qpos[2])) return 1;
        for (const auto& warning: data->warning) if (warning.number != 0) return 2;
    }
    return 0;
}
