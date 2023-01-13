#ifndef BCKENDFACTORY_HPP
#define BCKENDFACTORY_HPP

#include <map>
#include <string>
#include <iostream>
#include <memory>
#include "simpleLogger.hpp"



// Backend的注册
class Backend;
class BackendRegistry {
public:
    typedef std::shared_ptr<Backend > (*Creator)();
    typedef std::map<std::string, Creator> CreatorRegistry;

    static CreatorRegistry& Registry() {
        static CreatorRegistry* g_registry_ = new CreatorRegistry();
        return *g_registry_;
    }

    // Adds a creator.
    static void AddCreator(const std::string& type, Creator creator) {
        CreatorRegistry& registry = Registry();

        registry[type] = creator;
    }

    static std::shared_ptr<Backend > CreateBackend(std::string type) {
        CreatorRegistry& registry = Registry();

        return registry[type]();
    }

private:

    BackendRegistry() {}
};

class BackendRegisterer {
public:
    BackendRegisterer(const std::string& type,
                    std::shared_ptr<Backend> (*creator)() ) {
        BackendRegistry::AddCreator(type, creator);
    }
};

#define REGISTER_BACKEND_CREATOR(type, creator)                                  \
  static BackendRegisterer g_creator_f_##type(#type, creator);     \

#define REGISTER_BACKEND_CLASS(type)                                              \
  std::shared_ptr<Backend > Creator_##type##Backend() \
  {                                                                            \
    return std::shared_ptr<Backend>(new type##Backend());           \
  }                                                                            \
  REGISTER_BACKEND_CREATOR(type, Creator_##type##Backend)
#endif //BCKENDFACTORY_HPP
