# 进程内父600不可变缓存：未接入、未数值验证的最小原型

本原型服务未来I/O准备。**当前144的源码、PID、deadline、训练与资格不改；本工具不是恢复／执行授权，也不能接纳任何实际900检查点。** 原失败／deadline记录必须保留。

## 必须先证明的等价性

未来由父另设数值验证，uncached与cached严格加载路径必须在同一父41fa600、新600初始保存及未来900保存上核：state_dict全部值／dtype／shape、identity、canonical8Adam moments及计数、curve、原fcbd anchor、Torch／NumPy／Python RNG均exact；20actualX与12conditional各TF／预测target输出也须exact。缓存不得改变生产数学、参数、sourceconfig或新FORMAT。首miss必须真实调用原strictloader(final=True)，不得自造已验证标记。

重复hit必须证明所有返回模型／Adam／RNG／anchor与私有模板及先前返回值无可写Tensor storage共享，包括不同view重叠，不仅Python对象id不同；修改一个返回值不能影响下一返回。源、checkpoint或任意proof内容变动必须拒绝，不能用mtime／size／inode替代contentSHA，不能在失败后返回旧模板。别名model/common/sources/adapter与sys.path、RNG即使异常也必须恢复。缓存额外持有模型／payload可能增加RSS，原1800MiB门仍要验证。

## 原型边界和接口

`cache.py`只提供一个独立class，不修改任意父globals、不安装全局load wrapper、不接到现有train／verify／body。

- 调用方传入原strict_loader、原preserve_rng contextmanager，以及两个将来必须审核的检查函数：approve_parent负责不递归重构祖先的同一父模型／payload语义验证，assert_isolated负责完整Tensor storage隔离。原型不提供默认always-PASS检查函数；这是尚未完成的数值接入条件。
- 构造时指定固定checkpoint SHA／FORMAT、approved_ledger与runtime_identity。ledger必须包括checkpoint、所有祖先config source/data、原资格proof及本cache／未来adapter的源码SHA。只有checkpoint.config的一层映射不足以证明完整性；调用方必须从实际严格祖先调用链展开并审定完整ledger。原型不能自动从用户给的字典证明“无遗漏”。
- runtime_identity未来应由实际运行时读取Torch／NumPy／Python版本、dtype／CPU／线程和原loader实现身份，不能仅依赖调用方任填字符串。实例把这些与完整ledger组成key，禁止跨runtime复用。
- 每次get在入口逐内容SHA核全ledger；首miss进入原RNG保持上下文、调用原严格父600装载、检查原格式／600，再执行语义与隔离检查，把deepcopy模板私有持有，miss中途再次核ledger。
- hit仅deepcopy已验证模板，执行语义／隔离检查，出口再次全ledger SHA。重复hash仍可能昂贵，缓存只避免重复反序列化／祖先模型构造，不保证显著加速。finally恢复四alias／path，原preserve_rng包装器负责其实际定义域的RNG恢复；不克隆装饰器函数。
- check_original_anchor在已有严格miss后，按每个名字／shape／dtype／device／torch.equal逐值检查原anchor，不每次构造父链；不返回私有anchor引用。此函数仅为将来父端数值验证，子没有调用。

私有模板只是Python封装私有属性，不是硬件只读内存；调用方不得访问／修改_private，缓存实例不得跨进程传递。现有Python解释器中其他线程若改全局alias／RNG，实例内部RLock不足以保护它们；未来只在既有CPU1、无并发模型装载环境接入，或另证明进程级共享锁协议。内容校验前后也不构成对外部文件写者的原子锁；沿用严格加载前后SHA／冻结输入假设，若要求更强保证需独立冻结快照，不能声称已解决所有竞态。

## 与保存检查点来源身份的关系

缓存不会改原父checkpoint的sourceconfig，也不能把新cache源码插回旧冻结映射。未来独立I/O adapter必须有自己的ledger，绑定cache实现、完整原输入账本、原strict-loader源码和被加载checkpoint SHA；原config仍按原路径验证。新900模型文件本身如由未缓存源码训练产生，应保留该sourceconfig，用独立保存模型资格报告记录缓存路径等价证明，不重写权重文件或旧结果。若实际新loader改变initialize／source要求，必须另定义并验证兼容边界，不能以同一数学为由跳过原strict source identity。

## 若运行触deadline

当前不操作。如果未来确认900原子检查点、canonicalAdam计数900、曲线900与601..900完整训练记录齐全，但保存审核／EVAL触deadline，可另立postrun savedmodel资格恢复，只做严格装载／fresh重放／指标审核，optimizer.step和训练循环均为0。原JOBS FAIL／124和日志保留；没有900证据就不能声称完成900。该流程仍需父另立具体授权及数值验证，本原型不实现它。

## 当前验证状态

只做源码／标准库模拟fixture：用普通可变列表作为假model/payload验证首miss一次、hit深拷贝、返回值修改隔离、错误内容SHA拒绝及Python random状态恢复。它不替代Tensor storage隔离、strictNN装载、20／12输出一致性、版本／完整ledger审核或RSS／速度实测。实际模型验证尚未进行，禁止标缓存神经资格PASS。
