FROM golang:1.12

ADD main.go /main.go 

ADD cifar/ /cifar/

RUN export GOPATH=$HOME/go && cd / && go get -d -v .


