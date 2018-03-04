function Ein = evaluateError(X_data, W, Y_data)
Ein = sum(sign(X_data*W) ~= Y_data)/size(Y_data, 1);