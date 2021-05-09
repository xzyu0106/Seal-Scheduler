all: local

local:
	GOOS=linux GOARCH=amd64 go build  -o=seal-scheduler ./cmd/scheduler

build:
	sudo docker build --no-cache . -t registry.cn-hangzhou.aliyuncs.com/xzyu0106/seal-scheduler:1.3

push:
	sudo docker push registry.cn-hangzhou.aliyuncs.com/xzyu0106/seal-scheduler:1.3

format:
	sudo gofmt -l -w .
clean:
	sudo rm -f seal-scheduler
