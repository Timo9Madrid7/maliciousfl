# maliciousfl


## Running

`server.bat` for running server

`client.bat` for running client

*./Common/Utils/config.py* for setting configuration, where `num_workers` should be consistent to the number of clients

In each xx_client.py, modifying `if self.client_id < ` can add malicious clients

## grpc

### proto 设定

   ```c++
   service FL_Grpc
   {
       rpc UpdateIdx_uint32(IdxRequest_uint32) returns (IdxResponse_uint32){}
   }
   message IdxRequest_uint32
   {
     uint32 id = 1;
     repeated uint32 idx_ori = 2;
   }
   ```
   - `rpc 请求函数名 (参数) returns (返回值) {}`
   - `参数`用`message`类初始化, `uint32 id = 1`的解释为: 第`1`个字段的名称为`id`且它的类型为`uint32`
   - `repeated`代表可重复, 可以理解为数组 -- 该字段可以存放多组`uint32`

### 编译结果

    ```cmd
    python -m grpc_tools.protoc -I./ --python_out=./ --grpc_python_out=./ ./*.proto
    ```
    编译会生成*xxx_pb2_grpc.py* 和 *xxx_pb2.py*
    - *xxx_pb2.py*: request 和 response 的方法
    - *xxx_pb2_grpc.py*: client 与 server 类
      - `FL_GrpcServicer`: server 类
      - `FL_GrpcStub`: client 类
      - `add_FL_GrpcServicer_to_server`: 将对应的任务处理函数添加到rpc server中

## 多线程
`FlGrpcServer` 继承 `FL_GrpcServicer`, 其中 `process` 可多线程调用.
- `threading.Condition`: 本类用于实现条件变量对象. 条件变量对象允许多条线程保持等待状态直到接收另一条线程的通知.
- `acquire`: 获得lock
- `wait`: 等待被唤醒
- `notifyAll`: 唤醒所有线程
- `release`: 释放lock
- 上传由 client 端`update()`中`UpdateGrad_float`实现, 响应由服务端的`GradResponse_float`实现
