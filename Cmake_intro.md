# Cmake_intro

## 简介

构建 (build)：少量文件可以用几行命令完成编译，链接的过程，传统的 IDE 一般内置了构建系统。

而对于复杂项目，包含不同的模块，组件，每个组件由若干源文件组成及依赖的第三方库，此时就需要软件构建：

- 全自动完成代码编译，链接，打包的整个过程
- 管理不同组件，甚至包括第三方库的关联

**Cmake:** 是跨平台，开源的构建工具

## 流程

1.创建 `CMakeLists.txt` 文件

```cmake
# cmake 最低版本
cmake_minimum_required(3.10)
# 工程名称
project(project_name)
# 表示项目需要构建一个可执行文件,由 main.cpp 编译而成
add_executable(project_name main.cpp)
```

2.配置 (Configure)：根据  `CMakeLists.txt` 文件生成目标平台下的原生工程

选择平台原生的 `C++` 构建工具进行配置

3.构建 (Build)：使用 Cmake Build



























