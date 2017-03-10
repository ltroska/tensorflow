#include "tensorflow/hpx/core/distributed_runtime/hpx_master.h"

namespace tensorflow {

HPXMaster* NewHPXMaster(global_runtime* init, std::string const& basename)
{
  return new HPXMaster(init, basename);
}

}