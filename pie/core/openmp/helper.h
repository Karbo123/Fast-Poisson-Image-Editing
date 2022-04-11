#ifndef PIE_CORE_OPENMP_HELPER_H_
#define PIE_CORE_OPENMP_HELPER_H_

#include <tuple>

#include "solver.h"

class OpenMPEquSolver : public EquSolver {
  int* maskbuf;
  unsigned char* imgbuf;
  float* tmp;
  int n_mid;

 public:
  explicit OpenMPEquSolver(int n_cpu);
  ~OpenMPEquSolver();

  py::array_t<int> partition(py::array_t<int> mask);
  void post_reset();

  inline void update_equation(int i);

  void calc_error();

  std::tuple<py::array_t<unsigned char>, py::array_t<float>> step(
      int iteration);
};

class OpenMPGridSolver : public GridSolver {
  unsigned char* imgbuf;
  float* tmp;
  int m3;

 public:
  explicit OpenMPGridSolver(int grid_x, int grid_y, int n_cpu);
  ~OpenMPGridSolver();

  void post_reset();

  inline void update_equation(int id);

  void calc_error();

  std::tuple<py::array_t<unsigned char>, py::array_t<float>> step(
      int iteration);
};

#endif  // PIE_CORE_OPENMP_HELPER_H_
