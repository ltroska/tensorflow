#ifndef TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_TENSORFLOW_SERIALIZATION_H_
#define TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_TENSORFLOW_SERIALIZATION_H_

#include <hpx/include/serialization.hpp>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace hpx { namespace serialization {

inline void serialize(hpx::serialization::output_archive& ar, tensorflow::Status& s, unsigned)
{  
 ar << s.error_message() << s.code() ; 
}

inline void serialize(hpx::serialization::input_archive& ar, tensorflow::Status& s, unsigned)
{
  unsigned code_int;
  ar >> code_int;

  std::string error_msg;
  ar >> error_msg;
  
  if (code_int == 0)
    s = tensorflow::Status::OK();
  else
    s = tensorflow::Status(static_cast<tensorflow::error::Code>(code_int), error_msg);  
}

inline void serialize(hpx::serialization::output_archive& ar, const ::google::protobuf::Message& proto, unsigned v)
{
  std::string data;
  proto.SerializeToString(&data);
  
  ar << data;
}

inline void serialize(hpx::serialization::input_archive& ar, ::google::protobuf::Message& proto, unsigned v)
{
  std::string data;
  ar >> data;

  proto.ParseFromString(data);
}

inline void serialize(hpx::serialization::input_archive& ar, const ::google::protobuf::Message& proto, unsigned v)
{
  std::string data;
  ar >> data;
  
  const_cast<::google::protobuf::Message&>(proto).ParseFromString(data);
}

}}

#endif