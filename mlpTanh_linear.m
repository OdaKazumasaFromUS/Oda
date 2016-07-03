function [model, mse, Y] = mlpTanh_linear(X, Yhat, h)
% Multilayer perceptron
%�z��͏c�œ����
%PRML��5.3.2
%�B��w��tanh,�o�͑w�͐��`�̊������֐�
% Input:
%   X: d x n data matrix
%   Y: p x n response matrix
%   h: L x 1 vector specify number of hidden nodes in each layer l
% Ouput:
%   model: model structure
%   mse: mean square error
% Written by Mo Chen (sth4nth@gmail.com).
%�e���C���[�̃��j�b�g��
h = [size(X,1);h(:);size(Yhat,1)];
%���C���[��
L = numel(h);
W = cell(L-1);
for l = 1:L-1
    W{l} = randn(h(l),h(l+1));
end
Y = cell(L);
Y{1} = X;
%eta = 1/size(X,2);
eta = 0.01;
maxiter = 4000;
mse = zeros(1,maxiter);
mse(1) = 1;
%�`�揀��
figure(1)
plot(Yhat)
hold on
%iteration variable
iter = 1;
while mse(iter) > 0.0005
    iter = iter + 1;
    %forward
    %���ԑw�o��
    l = 2;
    %�������֐���`
    Y{l} = tanh(W{l-1}'*Y{l-1});
    
    %�o�͑w�o��
    l = 3;
    Y{l} = W{l-1}'*Y{l-1};
    
    %�o�͑w���j�b�g�ɑ΂���sigmaK
    sigK = Y{L} - Yhat;
    %     backward
    E = Yhat-Y{L};
    mse(iter) = mean(dot(E(:),E(:)));
    l = 2;
    %         df = (1-Y{l+1}.^2);
    %         dG = df.*E;
    %         dW = Y{l}*dG';
    
    sigJ = (1-Y{l}.^2).*(W{l}*sigK);
    
    W{1} = W{1}-eta*(sigJ*Y{1}')';
    W{2} = W{2}-eta*(sigK*Y{2}')';
    
    %     if mod(iter,10) == 0
    %         figure(1)
    %         plot(Y{L})
    %     end
    %�����̕`��
    figure(1)
    ylim([min(Yhat)*1.2, max(Yhat)*1.2])
    plot(Y{L})
    pause(.1)
end
mse = mse(1:iter);
model.W = W;
Y = Y{L};
disp(['iteration number:',num2str(iter)])
