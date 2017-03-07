#include "tensorflow/hpx/core/common_runtime/hpx_graph_runner.h"

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/cc/ops/standard_ops.h"

#include "tensorflow/core/graph/graph_constructor.h"

#include "tensorflow/cc/framework/scope.h"


int main(int argc, char* argv[])
{
  using namespace tensorflow;

  GraphDef graph_def;
  Status status = ReadBinaryProto(Env::Default(), "graph.pb", &graph_def);

  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }
      
  Scope root = Scope::NewRootScope();
    
  Graph* g = root.graph();

  status = ConvertGraphDefToGraph(GraphConstructorOptions(), graph_def, g);
    
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  Tensor a(DT_FLOAT, TensorShape());
  a.scalar<float>()() = 3.0;

  Tensor b(DT_FLOAT, TensorShape());
  b.scalar<float>()() = 5.0;  
  
  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
    { "a", a },
    { "b", b },
  };
    
  std::vector<tensorflow::Tensor> outputs;
  
  status = HPXGraphRunner::Run(g, nullptr, Env::Default(), inputs,
  {"c"}, &outputs);
      
  std::cout << outputs[0].DebugString() << std::endl;

  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }
     
  return 0;
}
