#ifndef TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_TENSORFLOW_SERIALIZATION_H_
#define TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_TENSORFLOW_SERIALIZATION_H_

#include <hpx/include/serialization.hpp>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/protobuf/worker.pb.h"
#include "tensorflow/core/distributed_runtime/call_options.h"
#include "grpc++/support/slice.h"

namespace hpx
{
namespace serialization
{
  inline void serialize(hpx::serialization::output_archive& ar,
                        tensorflow::Status& s,
                        unsigned)
  {
    ar << s.error_message() << s.code();
  }

  inline void serialize(hpx::serialization::input_archive& ar,
                        tensorflow::Status& s,
                        unsigned)
  {
    std::string error_msg;
    ar >> error_msg;

    unsigned code_int;
    ar >> code_int;

    if (code_int == 0)
      s = tensorflow::Status::OK();
    else
      s = tensorflow::Status(static_cast<tensorflow::error::Code>(code_int),
                             error_msg);
  }

  inline void serialize(hpx::serialization::output_archive& ar,
                        ::google::protobuf::Message& proto,
                        unsigned v)
  {
    std::string data;
    proto.SerializeToString(&data);

    ar << data;
  }

  inline void serialize(hpx::serialization::input_archive& ar,
                        ::google::protobuf::Message& proto,
                        unsigned v)
  {
    std::string data;
    ar >> data;

    proto.ParseFromString(data);
  }

  inline void serialize(hpx::serialization::output_archive& ar,
                        std::vector< ::grpc::Slice>& data,
                        unsigned v)
  {
    std::size_t num_slices = data.size();

    ar << num_slices;

    for (auto const& s : data)
      ar << s.size() << hpx::serialization::make_array(s.begin(), s.size());
  }

  inline void serialize(hpx::serialization::input_archive& ar,
                        std::vector< ::grpc::Slice>& buf,
                        unsigned v)
  {
    std::size_t num_slices;

    ar >> num_slices;

    buf.reserve(num_slices);

    for (std::size_t i = 0; i < num_slices; ++i) {
      std::size_t slice_len;
      ar >> slice_len;

      gpr_slice s = gpr_slice_malloc(slice_len);
      hpx::serialization::array<uint8_t> array =
          hpx::serialization::make_array(GPR_SLICE_START_PTR(s), slice_len);
      ar >> array;
      buf.push_back(::grpc::Slice(s, ::grpc::Slice::STEAL_REF));
    }
  }

  inline void serialize(hpx::serialization::output_archive& ar,
                        tensorflow::CallOptions opt,
                        unsigned v)
  {
    ar << opt.GetTimeout();
  }

  inline void serialize(hpx::serialization::input_archive& ar,
                        tensorflow::CallOptions opt,
                        unsigned v)
  {
    tensorflow::int64 timeout;
    ar >> timeout;

    opt.SetTimeout(timeout);
  }
}
}

#endif