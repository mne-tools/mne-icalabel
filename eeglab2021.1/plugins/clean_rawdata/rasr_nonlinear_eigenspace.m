function [Xsol, S0] = rasr_nonlinear_eigenspace(L, k, alpha)
% Example of nonlinear eigenvalue problem: total energy minimization.
%
% L is a discrete Laplacian operator: the covariance matrix
% alpha is a given constant for optimization problem
% k determines how many eigenvalues are returned 
%
% This example is motivated in the paper
% "A Riemannian Newton Algorithm for Nonlinear Eigenvalue Problems",
% Zhi Zhao, Zheng-Jian Bai, and Xiao-Qing Jin,
% SIAM Journal on Matrix Analysis and Applications, 36(2), 752-774, 2015.
%


% This file is part of Manopt and is copyrighted. See the license file.
%
% Main author: Bamdev Mishra, June 19, 2015.
% Contributors:
% Sarah Blum, 8/2018: changed the function to be included in Riemannian ASR:
%   additional outputs are needed: namely the eigenvectors and eigenvalues 
   
    n = size(L, 1);
    assert(size(L, 2) == n, 'L must be square.');
    
    if ~exist('alpha', 'var') || isempty(alpha)
        alpha = 1;
    end
        
    % Grassmann manifold description
    Gr = grassmannfactory(n, k);
    problem.M = Gr;
    
    % Cost function evaluation
    problem.cost =  @cost;
    function val = cost(X)
        rhoX = sum(X.^2, 2); % diag(X*X'); 
        val = 0.5*trace(X'*(L*X)) + (alpha/4)*(rhoX'*(L\rhoX));
    end
    
    % Euclidean gradient evaluation
    % Note: Manopt automatically converts it to the Riemannian counterpart.
    problem.egrad = @egrad;
    function g = egrad(X)
        rhoX = sum(X.^2, 2); % diag(X*X');
        g = L*X + alpha*diag(L\rhoX)*X;
    end
    
    % Euclidean Hessian evaluation
    % Note: Manopt automatically converts it to the Riemannian counterpart.
    problem.ehess = @ehess;
    function h = ehess(X, U)
        rhoX = sum(X.^2, 2); %diag(X*X');
        rhoXdot = 2*sum(X.*U, 2); 
        h = L*U + alpha*diag(L\rhoXdot)*X + alpha*diag(L\rhoX)*U;
    end
      
    % Initialization as suggested in above referenced paper.
    % randomly generate starting point for svd
    X = randn(n, k);
    [U, S, V] = svd(X, 0); %#ok<ASGLU>
    X = U*V';
    [U0, S0, ~] = eigs(L + alpha*diag(L\(sum(X.^2, 2))), k); %,'sm'); %#ok<NASGU,ASGLU>
    X0 = U0;
  
    % Call manoptsolve to automatically call an appropriate solver.
    % Note: it calls the trust regions solver as we have all the required
    % ingredients, namely, gradient and Hessian, information.
    Xsol = manoptsolve(problem, X0);
end
