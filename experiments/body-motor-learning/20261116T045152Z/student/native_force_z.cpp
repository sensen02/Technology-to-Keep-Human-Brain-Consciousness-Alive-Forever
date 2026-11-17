#include <mujoco/mujoco.h>

extern "C" double ground_force_z(const mjModel* model, const mjData* data) {
    double total = 0.;
    for (int index = 0; index < data->ncon; ++index) {
        const auto& contact = data->contact[index];
        const int first = contact.geom[0];
        const int second = contact.geom[1];
        if (first < 0 || second < 0 || contact.efc_address < 0) continue;
        const bool first_plane = model->geom_type[first] == mjGEOM_PLANE;
        const bool second_plane = model->geom_type[second] == mjGEOM_PLANE;
        if (!first_plane && !second_plane) continue;
        double force[6] = {};
        mj_contactForce(model, data, index, force);
        const double z = contact.frame[2] * force[0] + contact.frame[5] * force[1] + contact.frame[8] * force[2];
        total += (first_plane ? z : -z);
    }
    return total;
}
