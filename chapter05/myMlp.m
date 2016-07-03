%�O�w�j���[�����l�b�g
function [model, cost, Y] = myMlp(X, Yhat, h)
% Multilayer perceptron
% Input:
%   X: d x n data matrix
%   Yhat: p x n response matrix
%   h: L x 1 vector specify number of hidden nodes in each layer l
% Ouput:
%   model: model structure
%   cost: mean square error
%   Z:resultant data
% Written by Mo Chen (sth4nth@gmail.com).
X = X';
h = [size(X,1);h(:);size(Yhat,1)];
L = numel(h);
W = cell(L-1);
%�e�w�̏d�݌W��
for l = 1:L-1
    W{l} = randn(h(l),h(l+1));
end
%�e�w�̏o��
Y = cell(L);
%Z = cell(L);
%���͑w
Y{1} = X;
%�w�K�p�����[�^�H���O�Y�ꂽ
%eta = 1/size(X,2);
eta = 0.1;
%�ő�J��Ԃ��񐔁B��X�덷�֐��̃X���b�V�����h�ŏI������悤�ɕς���B
maxiter = 2000;
cost = zeros(1,maxiter);
cost(1) = 1;
iter = 1;
while cost(iter) > 0.05
%for iter = 1:maxiter
iter = iter + 1;
%     forward
    for l = 2:L-1
        Y{l} = sigmoid(W{l-1}'*Y{l-1});
    end
    
    for l = L
        Y{l} = W{l-1}'*Y{l-1};
    end
%     backward
    %�o�͑w�̏o��Y(L)�Ƌ��tYhat�̌덷�x�N�g��
    E = Y{L} - Yhat;
    %�W���X�V
    %�o�͑w
    for l = L-1
        df = ones(1,length(Yhat));
        deltaKout = (Y{l+1}-Yhat);
        dG = df.*E';
        %del(E)/del(w)
        dW = Y{l}*deltaKout';
        %���ԑw����o�͑w�ւ̏d�ݍX�V
        W{l} = W{l}-eta*dW;
        %���ԑw�ł̉��z�덷�֐�
        E = W{l}*dG';
    end
    %���ԑw
    for l = L-2:-1:1
        %del(Phi(vj))/del(vj)
        df = Y{l+1}.*(1-Y{l+1});
        %
        dG = df.*E;
        %del(E)/del(w)
        dW = Y{l}*dG';
        %���ԑw����o�͑w�ւ̏d�ݍX�V
        W{l} = W{l}-eta*dW;
        E = W{l}*dG;
    end
    cost(iter) = sum((Y{L} - Yhat).^2)/2;
end
cost = cost(1:iter);
model.W = W;
Y = Y{L};