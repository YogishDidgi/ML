function dataNew = mapminmax_mine(data, minVal, maxVal)
dataMax = max(data);
dataMin = min(data);

yMax = maxVal;
yMin = minVal;
xMax = dataMax;
xMin = dataMin;

slope = (yMax - yMin)/(xMax - xMin);
intercept = yMin;

dataNew = intercept + slope*(data - xMin);
end