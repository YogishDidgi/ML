function newW = linearRegression(X_data_Augment, Y_data, varargin)
if nargin == 2
    lambda = 0;
else
    lambda = varargin{1};
end

X = X_data_Augment;
X_T = X_data_Augment';
X_plus = pinv(X_T*X + lambda*eye(size(X_T*X)))*X_T;
newW = X_plus*Y_data;