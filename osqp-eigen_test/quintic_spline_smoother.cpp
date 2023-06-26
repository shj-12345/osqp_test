/**
 * @file quintic_spline_smoother.cpp
 * @author Giulio Romualdi
 * @copyright Released under the terms of the BSD 3-Clause License
 * @date 2018
 */
#include "matplotlibcpp.h"
#include "OsqpEigen/OsqpEigen.h"
#include <Eigen/Dense>
#include <iostream>
#include <math.h>
#include <vector>
namespace plt = matplotlibcpp;

#define X_ 0
#define Y_ 1

const int point_num = 20;
const int line_seg_num = point_num - 1;
const int variable_num = 6*line_seg_num;
const int middle_point_num = point_num - 2;
const int constraint_num = 5* middle_point_num + 8; 
//points 
Eigen::Matrix<double,point_num,3> point_xyt;

void setHessian(Eigen::SparseMatrix<double> &hessian)
{
  Eigen::Matrix<double,variable_num,variable_num> hessian_common;
  hessian_common.setZero();
  double dt;
  for(int i=0;i<line_seg_num;i++)
  {
    dt = point_xyt(i+1,2);
    hessian_common.block(6*i,6*i,6,6)<<
    0,0,0,0,0,0,
    0,0,0,0,0,0,
    0,0,0,0,0,0,
    0,0,0,36*dt,72*pow(dt,2),120*pow(dt,3),
    0,0,0,72*pow(dt,2),192*pow(dt,3),360*pow(dt,4),
    0,0,0,120*pow(dt,3),260*pow(dt,4),720*pow(dt,5);
  }  
  std::cout<<"hessian common:"<<std::endl;
  std::cout<<hessian_common.block(0,0,12,12)<<std::endl;
  // hessian.resize(variable_num,variable_num);
  hessian.resize(variable_num,variable_num);
  for(int i=0;i<hessian_common.rows();i++)
  {
    for(int j=0;j<hessian_common.cols();j++)
    {
      if(hessian_common(i,j)!=0)
      {
        hessian.insert(i,j) = hessian_common(i,j);
      }
    }
  }
  std::cout<<"***********"<<std::endl;
}

void setGradient(Eigen::VectorXd &gradient)
{
  gradient.resize(variable_num);
  for(int i=0;i<gradient.size();i++)
  {
    gradient(i)=0;
  }  
  std::cout<<"***********"<<std::endl;
}

void setLinearConstraint(Eigen::SparseMatrix<double> &LinearConstraint,Eigen::VectorXd &LowerBound,Eigen::VectorXd &UpperBound,int xy_choose)
{
  //中间点的连续性约束
  Eigen::Matrix<double,4*middle_point_num,variable_num> A1;
  A1.setZero();
  Eigen::VectorXd b1;
  b1.resize(4*middle_point_num);
  b1.setZero();
  double dt;
  for(int i=0;i<middle_point_num;i++)
  {
    dt=point_xyt(i+1,2);
    A1.block(4*i,6*i,4,12)<<
    1,dt,pow(dt,2),pow(dt,3),pow(dt,4),pow(dt,5),-1,0,0,0,0,0,
    0,1,2*dt,3*pow(dt,2),4*pow(dt,3),5*pow(dt,4),0,-1,0,0,0,0,
    0,0,2,6*dt,12*pow(dt,2),20*pow(dt,3),0,0,-2,0,0,0,
    0,0,0,6,24*dt,60*pow(dt,2),0,0,0,-6,0,0;
  }
  std::cout<<"***********"<<std::endl;

  //所有点的位置约束
  Eigen::Matrix<double,point_num,variable_num> A2;
  A2.setZero();
  Eigen::VectorXd b2;
  b2.resize(point_num);
  b2.setZero();
  for(int i=0;i<point_num;i++)
  {
    if(i<point_num-1)
    {
      A2.block(i,6*i,1,6)<<1,0,0,0,0,0;
    }
    else 
    {
      dt = point_xyt(i,2);
      A2.block(i,6*(i-1),1,6)<<1,dt,pow(dt,2),pow(dt,3),pow(dt,4),pow(dt,5);
    }
    b2(i)=point_xyt(i,xy_choose);//x=x(t)
  }
  std::cout<<b2.transpose()<<std::endl;
  std::cout<<"***********"<<std::endl;

  //起终点的导数约束
  Eigen::Matrix<double,6,variable_num> A3;
  A3.setZero();
  Eigen::VectorXd b3;
  b3.resize(6);
  b3.setZero();

  A3.block(0,0,3,6)<<
  0,1,0,0,0,0,
  0,0,2,0,0,0,
  0,0,0,6,0,0;
  b3.block(0,0,3,1)<<10,1,1;

  dt = point_xyt(point_num-1,2);
  A3.block(3,variable_num-6,3,6)<<
  0,1,2*dt,3*pow(dt,2),4*pow(dt,3),5*pow(dt,4),
  0,0,2,6*dt,12*pow(dt,2),20*pow(dt,3),
  0,0,0,6,24*dt,60*pow(dt,2);
  b3.block(3,0,3,1)<<0,0,0;
  std::cout<<"***********"<<std::endl;

  //矩阵和向量的拼接
  Eigen::Matrix<double,A1.rows()+A2.rows()+A3.rows(),variable_num> A;
  A << A1,A2,A3;

  Eigen::VectorXd b;
  b.resize(b1.rows()+b2.rows()+b3.rows());
  b << b1,b2,b3;

  //赋值
  LinearConstraint.resize(A.rows(),A.cols());
  for(int i=0;i<A.rows();i++)
  {
    for(int j=0;j<A.cols();j++)
    {
      if(A(i,j)!=0)
      {
        LinearConstraint.insert(i,j) = A(i,j);
      }
    }
  }
  
  LowerBound.resize(b.rows(),b.cols());
  LowerBound << b;
  UpperBound.resize(b.rows(),b.cols());
  UpperBound << b;
}

std::vector<double> x_array,y_array;
void show_spline(Eigen::VectorXd QPSolution_X,Eigen::VectorXd QPSolution_Y,Eigen::Matrix<double,point_num,3> point_xyt)
{
  // std::cout<<"coffX="<<QPSolution_X.transpose()<<std::endl;
  // std::cout<<"coffY="<<QPSolution_Y.transpose()<<std::endl;
  // Prepare data.
  
  double ts,te,t,interval;
  Eigen::VectorXd coff_x,coff_y,tt;
  coff_x.resize(6);
  coff_y.resize(6);
  tt.resize(6);
  int k = 50;
  for(int i=0;i<line_seg_num;i++)
  {
    ts = 0;
    te = point_xyt(i+1,2);
    interval = (te-ts)/k;
    coff_x = QPSolution_X.block(6*i,0,6,1);
    coff_y = QPSolution_Y.block(6*i,0,6,1);
    for(int j=0;j<k;j++)
    {
      t=ts+j*interval;
      tt<< 1,t,pow(t,2),pow(t,3),pow(t,4),pow(t,5);
      x_array.push_back(tt.transpose()*coff_x);
      y_array.push_back(tt.transpose()*coff_y);
    }
  }
  

}

int main()
{
  clock_t time_start = clock();
  clock_t time_end = clock();
  time_start = clock();
  
  // block of size (p,q) ,starting at (i,j)。matrix.block(i,j,p,q); matrix.block<p,q>(i,j);
  //get points
  double x=0,y;
  std::vector<double> x_origin,y_origin;
  for(int i=0;i<point_num;i++)
  {
    y=x*x/3+sin(x/2);
    point_xyt(i,0)=x;
    point_xyt(i,1)=y;
    x_origin.push_back(x);
    y_origin.push_back(y);
    if(i==0)point_xyt(i,2)=0;
    else 
    {
      point_xyt(i,2)=sqrt(pow(point_xyt(i,0)-point_xyt(i-1,0),2)+pow(point_xyt(i,1)-point_xyt(i-1,1),2))/20;//与上一点的欧式距离
    }
    x=x+3;
  }


  // allocate QP problem matrices and vectores
  Eigen::SparseMatrix<double> hessian;
  Eigen::VectorXd gradient;
  Eigen::SparseMatrix<double> linearMatrix_X,linearMatrix_Y;
  Eigen::VectorXd lowerBound_X,lowerBound_Y;
  Eigen::VectorXd upperBound_X,upperBound_Y;
  
  setHessian(hessian);
  setGradient(gradient);
  setLinearConstraint(linearMatrix_X,lowerBound_X,upperBound_X,X_);
  setLinearConstraint(linearMatrix_Y,lowerBound_Y,upperBound_Y,Y_);
  // std::cout<<"hessian:"<<hessian<<std::endl;
  // instantiate the solver
  OsqpEigen::Solver solver_X,solver_Y;

  // settings
  //solver.settings()->setVerbosity(false);
  solver_X.settings()->setWarmStart(true);
  solver_X.data()->setNumberOfVariables(variable_num);// set the initial data of the QP solver
  solver_X.data()->setNumberOfConstraints(constraint_num);
  if(!solver_X.data()->setHessianMatrix(hessian)) return 1;
  if(!solver_X.data()->setGradient(gradient)) return 1;
  if(!solver_X.data()->setLinearConstraintsMatrix(linearMatrix_X)) return 1;//lowerBound=<Ax<=upperBound
  if(!solver_X.data()->setLowerBound(lowerBound_X)) return 1;
  if(!solver_X.data()->setUpperBound(upperBound_X)) return 1;
  if(!solver_X.initSolver()) return 1;// instantiate the solver

  // settings
  //solver.settings()->setVerbosity(false);
  solver_Y.settings()->setWarmStart(true);
  solver_Y.data()->setNumberOfVariables(variable_num);// set the initial data of the QP solver
  solver_Y.data()->setNumberOfConstraints(constraint_num);
  if(!solver_Y.data()->setHessianMatrix(hessian)) return 1;
  if(!solver_Y.data()->setGradient(gradient)) return 1;
  if(!solver_Y.data()->setLinearConstraintsMatrix(linearMatrix_Y)) return 1;//lowerBound=<Ax<=upperBound
  if(!solver_Y.data()->setLowerBound(lowerBound_Y)) return 1;
  if(!solver_Y.data()->setUpperBound(upperBound_Y)) return 1;
  if(!solver_Y.initSolver()) return 1;// instantiate the solver

  // controller input and QPSolution vector
  
  Eigen::VectorXd QPSolution_X,QPSolution_Y;

  // // number of iteration steps
  // int numberOfSteps = 50;
  
  for (int i = 0; i < 30; i++)
  {
    // solve the QP problem
    if(solver_X.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) return 1;
    if(solver_Y.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) return 1;
    // get the controller input
    QPSolution_X = solver_X.getSolution();
    QPSolution_Y = solver_Y.getSolution();
    // show spline
    // show_spline(QPSolution_X,QPSolution_Y,point_xyt);
  }
  time_end = clock();
  std::cout << "time use:" << 1000 * (time_end - time_start) / (double)CLOCKS_PER_SEC << "ms" << std::endl;
  plt::figure_size(1200, 780);
  plt::plot(x_origin,y_origin,"*");
  plt::plot(x_array, y_array,"--");
  plt::show();
  
  // plt::close();
  return 0;
}
