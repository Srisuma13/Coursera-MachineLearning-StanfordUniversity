function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
z=zeros(m);
z=X*theta;
predictions=sigmoid(z);
for j=1:size(predictions,2),
  J(j)=0;
  for i=1:m,
    J(j)+=(-y(i,j)*log(predictions(i,j))-((1-y(i,j))*log(1-predictions(i,j))));
end
J(j)=J(j)/m;

for j=1:size(X,2),
  sum=0;
  for i=1:m,
    sum+=(predictions(i)-y(i))*X(i,j);
  end
  grad(j)=sum/m;
end




% =============================================================

end
