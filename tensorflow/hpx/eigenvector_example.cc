#include "tensorflow/core/common_runtime/hpx_graph_runner.h"

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/cc/ops/standard_ops.h"

#include "tensorflow/core/graph/graph_constructor.h"

#include "tensorflow/cc/framework/scope.h"

tensorflow::GraphDef CreateGraphDef() {
  using namespace tensorflow;
  // TODO(jeff,opensource): This should really be a more interesting
  // computation.  Maybe turn this into an mnist model instead?
  Scope root = Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  // A = [3 2; -1 0].  Using Const<float> means the result will be a
  // float tensor even though the initializer has integers.
  auto a = Const<float>(root, {{3, 2}, {-1, 0}});

  // x = [1.0; 1.0]
  auto x = Const(root.WithOpName("x"), {{1.f}, {1.f}});

  // y = A * x
  auto y = MatMul(root.WithOpName("y"), a, x);

  // y2 = y.^2
  auto y2 = Square(root, y);

  // y2_sum = sum(y2).  Note that you can pass constants directly as
  // inputs.  Sum() will automatically create a Const node to hold the
  // 0 value.
  auto y2_sum = Sum(root, y2, 0);

  // y_norm = sqrt(y2_sum)
  auto y_norm = Sqrt(root, y2_sum);

  // y_normalized = y ./ y_norm
  Div(root.WithOpName("y_normalized"), y, y_norm);

  GraphDef def;
  TF_CHECK_OK(root.ToGraphDef(&def));

  return def;
}

int main(int argc, char* argv[])
{
    using namespace tensorflow;
    Status status;

    Scope root = Scope::NewRootScope();
    
    Graph* g = root.graph();

    status = ConvertGraphDefToGraph(GraphConstructorOptions(), CreateGraphDef(), g);
    
    if (!status.ok()) {
      std::cout << status.ToString() << "\n";
      return 1;
    }
    
    Tensor x(DT_FLOAT, TensorShape({2, 1}));
    auto x_m = x.tensor<float, 2>();
    
    x_m(0, 0) = 1;
    x_m(1, 0) = 1;

    // Iterations.
    std::vector<Tensor> outputs;
    for (int iter = 0; iter < 100; ++iter) {
      outputs.clear();

    TF_CHECK_OK(
      HPXGraphRunner::Run(g, nullptr, Env::Default(), {{"x", x}}, {"y:0", "y_normalized:0"}, &outputs));
      CHECK_EQ(size_t{2}, outputs.size());


      const Tensor& y = outputs[0];
      const Tensor& y_norm = outputs[1];
      // Print out lambda, x, and y.
      std::cout << "------------------\n" << x.DebugString() << "\n" << y.DebugString() << "\n" << y_norm.DebugString() << std::endl;
      // Copies y_normalized to x.
      x = y_norm;
    }
      
    if (!status.ok()) {
      std::cout << status.ToString() << "\n";
      return 1;
    }   
     
    return 0;
}