FROM golang:alpine AS build-env

RUN apk update && apk upgrade && \
    apk add --no-cache bash git openssh

COPY ./apiserver /apiserver/apiserver
COPY ./apiserver.yaml /apiserver/apiserver.yaml
COPY ./go.mod /apiserver/go.mod
COPY ./main.go /apiserver/main.go

WORKDIR /apiserver
RUN go build -o goapp

# final stage
FROM alpine
WORKDIR /apiserver
COPY --from=build-env /apiserver/goapp /apiserver/apiserver
COPY --from=build-env /apiserver/apiserver.yaml /apiserver/apiserver.yaml
RUN mkdir /apiserver/files
ENTRYPOINT /apiserver/apiserver
