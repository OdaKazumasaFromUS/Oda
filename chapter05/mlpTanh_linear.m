function [model, mse, Y] = mlpTanh_linear(X, Yhat, h)
% Multilayer perceptron
%配列は縦で入れる
%PRMLの5.3.2
%隠れ層はtanh,出力層は線形の活性化関数
% Input:
%   X: d x n data matrix
%   Y: p x n response matrix
%   h: L x 1 vector specify number of hidden nodes in each layer l
% Ouput:
%   model: model structure
%   mse: mean square error
% Written by Mo Chen (sth4nth@gmail.com).
%各レイヤーのユニット数
h = [size(X,1)+1;h(:)+1;size(Yhat,1)];
%レイヤー数
L = numel(h);
W = cell(L-1);
for l = 1:L-1
    W{l} = randn(h(l),h(l+1));
end
Y = cell(L);
Y{1} = [1;X];
%eta = 1/size(X,2);
eta = 0.01;
maxiter = 4000;
mse = zeros(1,maxiter);
mse(1) = 1;
%描画準備
figure(1)
plot(Yhat)
hold on
%iteration variable
iter = 1;
while mse(iter) > 0.0005
    iter = iter + 1;
    %forward
    %中間層出力
    l = 2;
    %活性化関数定義
    Y{l} = tanh(W{l-1}'*Y{l-1});
    
    %出力層出力
    l = 3;
    Y{l} = W{l-1}'*Y{l-1};
    
    %出力層ユニットに対するsigmaK
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
    %成長の描画
    figure(1)
    ylim([min(Yhat)*1.2, max(Yhat)*1.2])
    plot(Y{L})
    pause(.1)
end
mse = mse(1:iter);
model.W = W;
Y = Y{L};
disp(['iteration number:',num2str(iter)])

