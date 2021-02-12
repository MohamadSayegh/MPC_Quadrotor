# Model Predictive Control for a quadrotor in static and dynamic environments

KU Leuven

Optimization of Mechatronic Systems H04U1a

Authors: Elias Rached,Mohamad Sayegh

## Overview

Based on the research paper “Model Predictive Control for Aerial Collision Avoidance in dynamic environments”

Goal is to safely navigate in a workspace populated by static and moving obstacles

## Objectives

- Implement point-to-point Model Predictive Control

- Obstacle Avoidance in static and dynamic environment (Moving obstacles)

- Finding the best formulation of the problem ( MPC and Obstacles )

- Exploring time optimality (comparison several methods / parameters)

- How to escape Local minima

## Application 

- Using Rockit: Rockit (Rapid Optimal Control kit) is a software framework to quickly prototype optimal control problems (aka dynamic optimization) that may arise in     engineering: iterative learning (ILC), model predictive control (NMPC), motion planning. Link: https://github.com/meco-group/rockit 

- Solver used: IPOPT (Interior Point OPTimizer)

- Control input limits:      −𝟏 ≤𝑢 ≤𝟏

- OCP horizon length: N = 10 - 20 control intervals

- Sampling time: dt = 0.05 - 0.1s

- Discretization method: multiple shooting 

- Numerical integration method: Runge-Kutta 4th order

- Using warmstarting

## Dealing with Local minima 

Methods of escaping the local minimum are proposed:

- Obstacle merging: when obstacles are close enough they are merged into one obstacle

- A* trajectory guiding: when confirmed to be stuck in a local minimum, an A* trajectory is found, and intermediate points are extracted to guide the drone away

- Potential Field method could help escape certain local minima (if close to obstacles)

## Example Results

### Dynamic enviroment

![alt-text](https://github.com/MohamadSayegh/MPC_Quadrotor/blob/main/dynamic%20enciroment.gif)








