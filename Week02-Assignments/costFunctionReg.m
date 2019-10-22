function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
sumTh=0;
grad = zeros(size(theta));
predictions=sigmoid(X*theta);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
for j=2:size(theta),
  sumTh+=theta(j)^2;
end;

for j=1:size(predictions,2),
  J(j)=0;
  for i=1:m,
    J(j)+=(-y(i,j)*log(predictions(i,j))-((1-y(i,j))*log(1-predictions(i,j))));
  end;
  J(j)+=lambda*sumTh*0.5;
  J(j)/=m;
end;

for j=1:size(X,2),
  sum=0;
  for i=1:m,
    sum+=(predictions(i)-y(i))*X(i,j);
  end;
  if j!=1,
    sum+=lambda*theta(j);
  end;
  grad(j)=sum/m;
end


% =============================================================

end
