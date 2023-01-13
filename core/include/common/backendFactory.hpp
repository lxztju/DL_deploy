

#ifndef Backend_FACTORY_H_
#define Backend_FACTORY_H_


#include <map>
#include <string>
#include <iostream>
#include <memory>


class Backend;
class BackendRegistry {
 public:
  typedef std::shared_ptr<Backend> (*Creator)();
  typedef std::map<std::string, Creator> CreatorRegistry;

  static CreatorRegistry& Registry() {
    static CreatorRegistry* g_registry_ = new CreatorRegistry();
    return *g_registry_;
  }

  static void AddCreator(const std::string& type, Creator creator) {
    CreatorRegistry& registry = Registry();
    if (registry.count(type) != 0) {
    //   LOG(INFO) << "Backend type " << type << " already registered.";
    }
    registry[type] = creator;
  }

  static std::shared_ptr<Backend> CreateBackend(const std::string type) {
    CreatorRegistry& registry = Registry();
    if (registry.count(type) != 1) {
    //   LOG(ERROR) << "Backend type " << type << " haven't registered.";
    }
    return registry[type]();
  }
 private:
  BackendRegistry() {}

};

class BackendRegisterer {
 public:
  BackendRegisterer(const std::string& type,
                std::shared_ptr<Backend> (*creator)()) {
    // LOG(INFO) << "Registering Backend type: " << type;
    BackendRegistry::AddCreator(type, creator);
  }
};

#define REGISTER_BACKEND_CREATOR(type, creator)                                    \
  static BackendRegisterer g_creator_##type(#type, creator)

#define REGISTER_BACKEND_CLASS(type)                                               \
  std::shared_ptr<Backend> Creator_##type##Backend()                                   \
  {                                                                            \
    return std::shared_ptr<Backend>(new type##Backend());                              \
  }                                                                            \
  REGISTER_BACKEND_CREATOR(type, Creator_##type##Backend);                             \
//   static type##Backend type
#endif //Backend_FACTORY_H_