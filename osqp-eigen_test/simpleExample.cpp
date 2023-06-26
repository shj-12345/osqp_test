// osqp-eigen
#include "OsqpEigen/OsqpEigen.h"
#include <Eigen/Dense>
#include <iostream>
int main(){  
    Eigen::SparseMatrix<double> hessian;
    Eigen::VectorXd gradient;
    Eigen::SparseMatrix<double> linearMatrix;
    Eigen::VectorXd lowerBound;
    Eigen::VectorXd upperBound;
    
    Eigen::Matrix<double,2,2> hessian_common;
    hessian.resize(2, 2);
    //构造matrix
    hessian_common << 1,-1,
                     -1, 2;
    //通过insert(row,col)将matrix转换为sparseMatrix
    for(int i=0;i<2;i++)
    {
        for(int j=0;j<2;j++)
        {
            if(hessian_common(i,j)!=0)
            {
                hessian.insert(i,j)= hessian_common(i,j);
            }
        }
    }
    //判断矩阵是否
    Eigen::LLT<Eigen::MatrixXd> lltOfA(hessian_common); // compute the Cholesky decomposition of A 
    if(lltOfA.info() == Eigen::NumericalIssue) 
    { 
     throw std::runtime_error("Possibly non semi-positive definitie matrix!"); 
    }  

    std::cout << "hessian:" << std::endl
                << hessian << std::endl;
   

    

    gradient.resize(2);
    gradient << -2, -6;
    std::cout << "gradient:" << std::endl
                << gradient << std::endl;

    linearMatrix.resize(3, 2);
    linearMatrix.insert(0, 0) = 1;
    linearMatrix.insert(0, 1) = 1;
    linearMatrix.insert(1, 0) = -1;
    linearMatrix.insert(1, 1) = 2;
    linearMatrix.insert(2, 0) = 2;
    linearMatrix.insert(2, 1) = 1;
    std::cout << "linearMatrix:" << std::endl
                << linearMatrix << std::endl;

    lowerBound.resize(3);
    lowerBound << -OsqpEigen::INFTY, -OsqpEigen::INFTY, -OsqpEigen::INFTY;
    std::cout << "lowerBound:" << std::endl
                << lowerBound << std::endl;

    upperBound.resize(3);
    upperBound << 2, 2, 3;
    std::cout << "upperBound:" << std::endl
                << upperBound << std::endl;

    int NumberOfVariables = 2;   //A矩阵的列数
    int NumberOfConstraints = 3; //A矩阵的行数
    
    lowerBound << -OsqpEigen::INFTY, -OsqpEigen::INFTY, -OsqpEigen::INFTY;
    OsqpEigen::Solver solver;

    // settings
    //solver.settings()->setVerbosity(false);
    solver.settings()->setWarmStart(true);

    // set the initial data of the QP solver
    //矩阵A为m*n矩阵
    solver.data()->setNumberOfVariables(NumberOfVariables);     //设置A矩阵的列数，即n
    solver.data()->setNumberOfConstraints(NumberOfConstraints); //设置A矩阵的行数，即m
    if (!solver.data()->setHessianMatrix(hessian))
        return 1; //设置P矩阵
    if (!solver.data()->setGradient(gradient))
        return 1; //设置q or f矩阵。当没有时设置为全0向量
    if (!solver.data()->setLinearConstraintsMatrix(linearMatrix))
        return 1; //设置线性约束的A矩阵
    if (!solver.data()->setLowerBound(lowerBound))
        return 1; //设置下边界
    if (!solver.data()->setUpperBound(upperBound))
        return 1; //设置上边界

    // instantiate the solver
    if (!solver.initSolver())
        return 1;

    Eigen::VectorXd QPSolution;

    // solve the QP problem
    if (solver.solveProblem()!= OsqpEigen::ErrorExitFlag::NoError)
        return 1;

    // get the controller input
    clock_t time_start = clock();
    clock_t time_end = clock();
    time_start = clock();
    QPSolution = solver.getSolution();
    time_end = clock();
    std::cout << "time use:" << 1000 * (time_end - time_start) / (double)CLOCKS_PER_SEC << "ms" << std::endl;

    std::cout << "QPSolution:" << std::endl
                << QPSolution << std::endl;


  }
  