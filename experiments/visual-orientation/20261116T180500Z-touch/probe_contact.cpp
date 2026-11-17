#include <cmath>
#include <mujoco/mujoco.h>

extern "C" void inspect_contact(const mjModel* model, const mjData* data,
                                const int* leg, const int* environment, int probe,
                                double* out) {
  for (int i=0; i<6; ++i) out[i]=0.0;
  for (int i=0; i<data->ncon; ++i) {
    const mjContact& contact = data->contact[i];
    const int a = contact.geom[0], b = contact.geom[1];
    if (a<0 || b<0 || contact.efc_address<0) continue;
    const bool first = environment[a]!=0, second = environment[b]!=0;
    if (first == second) continue;
    const int fly = first ? b : a, other = first ? a : b;
    mjtNum force[6];
    mj_contactForce(model, data, i, force);
    if (force[0]<=0.0) continue;
    const int side = leg[fly];
    if (other==probe) {
      if (side>=0) out[2+side] += force[0];
      else out[4] += force[0];
    }
    if (side>=0 && std::abs(contact.frame[2])<.65)
      out[side] += force[0]*std::sqrt(1.0-contact.frame[2]*contact.frame[2]);
  }
  out[5] = data->time;
}
