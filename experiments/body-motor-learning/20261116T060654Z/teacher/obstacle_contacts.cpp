#include <algorithm>
#include <mujoco/mujoco.h>

// Each contact is attributed to the actual solved collision geometry.
extern "C" void obstacle_contacts(const mjModel* model, const mjData* data, const int obstacle,
                                  const int thorax, const int* sensor_geoms, double* out) {
    std::fill(out, out + 8, 0.);
    if (obstacle < 0) return;
    const auto* rotation = data->xmat + 9*thorax;
    double local[6];
    for (int i = 0; i < data->ncon; ++i) {
        const auto& contact = data->contact[i];
        const int a = contact.geom[0], b = contact.geom[1];
        if (contact.efc_address < 0 || (a != obstacle && b != obstacle)) continue;
        mj_contactForce(model, data, i, local);
        if (local[0] <= 0.) continue;
        const int body_geom = a == obstacle ? b : a;
        const double direction = a == obstacle ? 1. : -1.;
        double world[3];
        for (int axis = 0; axis < 3; ++axis) {
            world[axis] = direction*(contact.frame[axis]*local[0] + contact.frame[axis+3]*local[1]
                + contact.frame[axis+6]*local[2]);
            out[axis] += world[axis];
        }
        int segment = 0;
        while (segment < 3 && sensor_geoms[segment] != body_geom) ++segment;
        out[4+segment] += 1.;
        if (segment < 3) {
            const double parallel = world[0]*rotation[0] + world[1]*rotation[3] + world[2]*rotation[6];
            out[3] += std::max(0., -parallel);
        }
    }
}
