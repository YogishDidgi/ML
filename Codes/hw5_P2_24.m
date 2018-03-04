% function hw5_P2_24()
%%
close all;
numSamples = 10:10:10000;
E_Out = zeros(length(numSamples), 1);
bias = zeros(length(numSamples), 1);
var = zeros(length(numSamples), 1);
for i = 1:length(numSamples)
numIter = numSamples(i);
x = -1:0.01:1;
f_x = x.*x;

a = zeros(numIter, 1);
b = zeros(numIter, 1);

x1 = 2*rand(numIter, 1) - 1;
x2 = 2*rand(numIter, 1) - 1;
y1 = x1.*x1;
y2 = x2.*x2;
a = x1 + x2;
b = -(x1.*x2);
a_bar = mean(a);
b_bar = mean(b);
bias_x = x.^4 + (-2*a_bar)*(x.^3) + (a_bar*a_bar - 2*b_bar)*(x.^2) + (2*a_bar*b_bar)*x + (b_bar*b_bar);
bias(i) = mean(bias_x);

var_term1 = (a - a_bar).*(a - a_bar);
var_term1_mean = mean(var_term1);
var_term2 = 2*(a - a_bar).*(b - b_bar);
var_term2_mean = mean(var_term2);
var_term3 = (b - b_bar).*(b - b_bar);
var_term3_mean = mean(var_term3);

var_x = var_term1_mean*(x.^2) + var_term2_mean*x + var_term3_mean;
var(i) = mean(var_x);

E_Out(i) = bias(i) + var(i);

gbar_x = a_bar*x + b_bar;
% plot(x, gbar_x, '+r');hold on
% plot(x, f_x, '-b');hold off

end
figure, plot(numSamples, E_Out, numSamples, bias, numSamples, var);
legend('EOut','bias','var');
xlabel('X');
%%
a_mean = 0;
b_mean = 0;
for i = 1:numIter
    x1 = rand()*2 - 1;
    x2 = rand()*2 - 1;
%     y1 = x1*x1;
%     y2 = x2*x2;
    g_x = (x1 + x2)*x - x1*x2;
    a(i) = (x1 + x2);
    b(i) = (-x1*x2);
    a_mean = a_mean + (x1 + x2);
    b_mean = b_mean - x1*x2;
    plot(x, g_x, '-r');hold on;
end
a_mean = a_mean/numIter;
b_mean = b_mean/numIter;
gbar_x = a_mean*x + b_mean;

disp(a_mean);
disp(b_mean);
plot(x, f_x, '*b');
hold off;


%bias, variance, Eout
bias_x = (gbar_x - f_x).*(gbar_x - f_x);
bias = mean(bias_x)
var = 0;
for i = 1:numIter
    g_x = a(i)*x + b(i);
    var = var + (g_x - gbar_x).*(g_x - gbar_x);
end
var = mean(var);

Eout = bias + var;
