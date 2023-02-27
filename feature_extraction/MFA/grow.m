% C = grow(C,G, alpha)
%
% UPDATE ME!
% 
% 'Grow' a community C by adding to it all such vertices v, such that
% adding v to C does not decrease the density of C by too much.
% 'Too much' is defined in terms of the parameter alpha.
%
% Author: Jeff Noel
% Created:  Sept.2, 2007
% Modified: Oct.7, 2008


function [C,R] = grow(C,G,R, priority, lambda, t)

% Check validity of inputs:
[one n]= size(C);
if one ~= 1
    error('Input C to "grow" should be a row vector');
end
if length(lambda) ~= 1
    error('Input lambda to "grow" should be a scalar');
end
if length(t) ~= 1
    error('Input t to "grow" should be a scalar');
end
if any( size(G) ~= [n n] )
    error('Dimension mismatch in inputs of "grow"');
end
if any( size(priority) ~= [n n] )
    error('Dimension mismatch in inputs of "grow"');
end
 % Priority is the weight matrix?
sizeOfC = sum(C);
C_complement = (C == 0);
%d = density(C,G);

%update by Qi 6/21/2009
d=sum((C * G) .* C)/(sizeOfC*(sizeOfC-1));

% CHECK: these matrix ops should accomplish the same thing as the loops below.

% contribution(v) := sum of edge weights from v to C
contribution = (C * G) .* C_complement;

% Potential used to break ties in 'contribution'.
% For vertex v not in C, it is the sum of priorities of (u,v)s over all u in C.
potential = (C * priority) .* C_complement;

% for i = 1 : n
%    if C(i) == 0 % vertex i is not in C
%        contribution(i) = C * G(:,i);
%        potential(i) = C * priority(:,i);
%    else
%        contribution(i) = 0;
%        potential(i) = 0;
%    end
% end
    
% largest contribution scaled to 
maxContribution = max( contribution );
alpha_n = 1 - 1/(2*lambda*(sizeOfC+t));
q=1;
while maxContribution > alpha_n * d * sizeOfC % by *sizeofC so that we can compare the numerator 
    % find the vertex with maximal contribution,
    % using potential to break ties
    maximalElements = (contribution == maxContribution);
    [bestPotential v] = max( maximalElements .* potential );
                
    % add v to C and update density and contribution accordingly.
    C(v) = 1;
    C_complement(v) = 0;
    R(v)=q+1;
    q=q+1;

    % Reminder: 2 comes from the n-choose-2 in the max edge-set size.
    d = (d*(sizeOfC-1) + 2*maxContribution/sizeOfC )/(sizeOfC+1);
    sizeOfC = sizeOfC+1;      
    alpha_n = 1 - 1/(2*lambda*(sizeOfC+t));
   
    contribution = contribution + G(v,:) .* C_complement;
    contribution(v) = 0; 

    % for j=1 : verticesRemaining
    %     u = contribution(j,1);
    %     contribution(j,2) = (contribution(j,2) * (verticesRemaining-1) ...
    %         + G(u,v)) / verticesRemaining;
    % end        
    maxContribution = max( contribution );
end
