%三層ニューラルネット
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
%各層の重み係数
for l = 1:L-1
    W{l} = randn(h(l),h(l+1));
end
%各層の出力
Y = cell(L);
%Z = cell(L);
%入力層
Y{1} = X;
%学習パラメータ？名前忘れた
%eta = 1/size(X,2);
eta = 0.1;
%最大繰り返し回数。後々誤差関数のスレッショルドで終了するように変える。
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
    %出力層の出力Y(L)と教師Yhatの誤差ベクトル
    E = Y{L} - Yhat;
    %係数更新
    %出力層
    for l = L-1
        df = ones(1,length(Yhat));
        deltaKout = (Y{l+1}-Yhat);
        dG = df.*E';
        %del(E)/del(w)
        dW = Y{l}*deltaKout';
        %中間層から出力層への重み更新
        W{l} = W{l}-eta*dW;
        %中間層での仮想誤差関数
        E = W{l}*dG';
    end
    %中間層
    for l = L-2:-1:1
        %del(Phi(vj))/del(vj)
        df = Y{l+1}.*(1-Y{l+1});
        %
        dG = df.*E;
        %del(E)/del(w)
        dW = Y{l}*dG';
        %中間層から出力層への重み更新
        W{l} = W{l}-eta*dW;
        E = W{l}*dG;
    end
    cost(iter) = sum((Y{L} - Yhat).^2)/2;
end
cost = cost(1:iter);
model.W = W;
Y = Y{L};