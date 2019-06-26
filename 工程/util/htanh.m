function P = htanh(P)
    P(find(P>=0))=1;
    P(find(P<0))=-1;
end