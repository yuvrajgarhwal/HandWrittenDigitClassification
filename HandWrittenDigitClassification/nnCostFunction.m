function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)


Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


m = size(X, 1);
         

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
X = [ones(m, 1) X];
z2=X*Theta1';
a2=sigmoid(z2);
a2=[ones(size(a2,1),1) a2];
z3=a2*Theta2';
H=sigmoid(z3);
y_matrix = eye(num_labels)(y,:);
J=(-1/m)*(sum(sum(y_matrix.*log(H)+(1.-y_matrix).*log(1-H))));

temp=sum(sum(Theta1(:,(2:end)).^2))+sum(sum(Theta2(:,(2:end)).^2));
J=J+(lambda/(2*m))*temp;


delta1=zeros(size(Theta1));
delta2=zeros(size(Theta2));
for t=1:m
  a1t=X(t,:);
  a2t=a2(t,:);
  a3t=H(t,:);
  d3=a3t-y_matrix(t,:);
  d2=Theta2'*d3'.* sigmoidGradient([1;Theta1*a1t']);
  delta1=delta1+d2(2:end)*a1t;
  delta2=delta2+d3'*a2t;
endfor
  Theta1_grad=(1/m)*delta1;
  Theta2_grad=(1/m)*delta2;

  Theta1_grad=Theta1_grad+(lambda/m)*[zeros(size(Theta1,1),1) Theta1(:,2:end)];
  Theta2_grad=Theta2_grad+(lambda/m)*[zeros(size(Theta2,1),1) Theta2(:,2:end)];

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
