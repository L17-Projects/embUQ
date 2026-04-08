# Directory Guide

## Purpose
Core hierarchical-problem implementation shared by `Hierarchical/Psi`, `Hierarchical/Theta`, and related Korali modules.

## Specifics
This directory contains the common hierarchical base class plus concrete hierarchical problem implementations and their generated-source inputs (`*.base`, `*.config`). Changes here affect Korali's internal hierarchical inference behavior and configuration serialization.

## Provenance
Hand-maintained C++ module sources with generated-code templates committed alongside the emitted files.

## Immediate Contents
- Subdirectories: concrete hierarchical problem modules such as `psi`, `theta`, and `thetaNew`.
- Files: shared base implementation files such as `hierarchical.cpp`, `hierarchical.hpp`, and `hierarchical.config`.

## Maintenance
Update this file whenever the directory's role or meaningful contents change.
