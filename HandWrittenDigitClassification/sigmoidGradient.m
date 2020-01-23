function g = sigmoidGradient(z)

g = zeros(size(z));




temp=sigmoid(z);
g=temp.*(1.-temp);









end
