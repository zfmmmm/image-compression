# ==============================================================================
# decode.py - 统一格式批量并行解码器
# ==============================================================================
import argparse  # 导入标准库 argparse，用于解析命令行输入的参数
import logging  # 导入标准库 logging，用于记录程序的运行状态、警告和错误信息
from pathlib import Path  # 导入标准库 Path，提供面向对象的文件系统路径操作，比 os.path 更优雅且跨平台
from concurrent.futures import ThreadPoolExecutor, as_completed  # 导入并发库，提供线程池和任务完成状态监控
import sys  # 导入标准库 sys，用于在程序遇到严重错误时执行 sys.exit() 以非零状态码退出程序

from tqdm import tqdm  # 导入第三方库 tqdm，用于在控制台渲染直观的进度条

# 导入项目特有的路径兼容性补丁函数
from utils.codec_imports import ensure_project_codecs_importable  # 导入补丁函数，解决本地 codecs 目录与 Python 标准库 codecs 冲突的问题
ensure_project_codecs_importable()  # 立即执行该函数，将本地的 codecs 目录优先级提升，确保后续 from codecs.xxx 能够正确映射到本项目文件

# 导入项目内部核心模块
from codecs.base import BaseCodec  # 导入所有编码器的基类 BaseCodec，用于类型提示和约束
from utils.codec_registry import CODEC_CLASSES  # 导入编码器注册表字典，包含了字符串名称到具体编码器类（如 JXLCodec）的映射
from utils.logging_utils import setup_logging  # 导入项目自定义的日志初始化函数，用于配置日志格式和输出级别

LOGGER = logging.getLogger(__name__)  # 初始化当前模块的全局日志记录器，__name__ 会自动被替换为当前模块名（__main__ 或 decode）


def parse_args() -> argparse.Namespace:
    """
    定义并解析命令行参数。
    作用：提供灵活的用户交互接口，决定输入源、输出地、并发数以及外部命令超时机制。
    """
    # 实例化参数解析器，description 会在用户输入 --help 时显示为程序的整体说明
    parser = argparse.ArgumentParser(description="并行批量统一解码工具 (支持 JPEG, JPEG2000, JXL, AVIF, BPG)")
    
    # 定义 --input 参数：必须提供，类型限制为 Path 对象。既可以是单个压缩文件，也可以是包含多种压缩文件的目录
    parser.add_argument("--input", required=True, type=Path, help="输入路径 (单个压缩文件或包含压缩文件的目录)")
    
    # 定义 --output-dir 参数：必须提供，类型为 Path。解码后的 .png 文件将统一存放在这里
    parser.add_argument("--output-dir", required=True, type=Path, help="输出目录 (保存重建的PNG图像)")
    
    # 定义 --workers 参数：非必须，类型为整数。如果不填，将交由 ThreadPoolExecutor 默认根据系统 CPU 核心数自动计算最佳线程数
    parser.add_argument("--workers", type=int, default=None, help="并发解码的线程数 (默认自动根据CPU核心数推断)")
    
    # 定义 --timeout 参数：类型为浮点数。防止外部解码工具（如 avifdec）在处理损坏文件时死锁卡住，默认 600 秒强制杀掉进程
    parser.add_argument("--timeout", type=float, default=600.0, help="单个外部解码命令的超时时间 (秒)")
    
    # 定义 --log-level 参数：用于控制日志信息的冗余度，默认 INFO。调试时可设为 DEBUG
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志输出级别")
    
    return parser.parse_args()  # 解析用户在命令行实际输入的字符串，返回一个包含所有参数属性的 Namespace 对象


def build_decoder_routing_table(timeout: float) -> dict[str, BaseCodec]:
    """
    构建扩展名到解码器实例的路由表。
    作用：自动探测当前系统中安装了哪些底层解码工具，并将支持的扩展名映射到对应的 Python 封装类实例上。
    """
    routing_table: dict[str, BaseCodec] = {}  # 初始化空字典，键为扩展名（如 '.jxl'），值为对应的解码器实例
    
    # CODEC_CLASSES 包含别名（如 'jpeg2000' 和 'jp2' 指向同一个类），所以使用 set() 去重，避免重复实例化
    unique_codec_classes = set(CODEC_CLASSES.values())
    
    # 遍历每一个不重复的编码器类
    for codec_cls in unique_codec_classes:
        # 实例化该编码器，传入超时时间，确保其实例方法可用
        codec_instance = codec_cls(timeout=timeout)
        
        # 调用实例自带的可用性检测方法（该方法内部通常会用 shutil.which 检测外部命令是否存在）
        if codec_instance.is_available():
            # 将该解码器声明的特定扩展名转为小写作为 Key，实例作为 Value 存入路由表
            routing_table[codec_instance.bitstream_extension.lower()] = codec_instance
        else:
            # 如果底层工具缺失，在日志中记录一条警告，提示用户哪些格式将无法被解码
            LOGGER.warning("解码器 %s 的底层工具未安装，相关格式将被跳过", codec_instance.name)
            
    return routing_table  # 返回构建完毕的路由字典


def decode_single_file(input_file: Path, output_file: Path, codec: BaseCodec) -> tuple[Path, bool, str]:
    """
    执行单个文件的解码任务。
    此函数将在独立的线程中运行，因此必须捕获所有异常并返回状态，而不是直接抛出导致线程崩溃。
    """
    try:
        # 调用具体的解码器实例的 decode 方法。内部会启动 subprocess 运行相应的 C/C++ 命令行工具
        codec.decode(input_file, output_file)
        # 如果没有抛出异常，说明解码成功，返回 (文件路径, 成功标志 True, 提示信息 "OK")
        return (input_file, True, "OK")
    except Exception as exc:
        # 捕获任何潜在的异常（如 subprocess 超时、文件损坏等）
        error_msg = str(exc)  # 将异常对象转换为字符串描述
        # 记录详细的错误日志
        LOGGER.error("解码失败 [%s]: %s", input_file.name, error_msg)
        # 解码失败，返回 (文件路径, 失败标志 False, 具体的错误信息)
        return (input_file, False, error_msg)


def main() -> int:
    """
    程序的主入口点。
    负责协调参数解析、路由表构建、文件扫描、多线程任务分发以及结果统计。
    """
    args = parse_args()  # 调用解析函数，获取带有用户指定属性的 args 对象
    setup_logging(args.log_level)  # 根据用户指定的日志级别，配置全局日志系统
    
    # 检查用户提供的输入路径是否存在，不存在则拦截并报错
    if not args.input.exists():
        LOGGER.error("输入路径不存在: %s", args.input)  # 记录错误信息
        return 2  # 返回非 0 状态码 2，符合 Unix 退出码规范（表示命令行用法错误或输入文件缺失）

    # 动态构建解码路由表，只有安装了底层依赖的解码器才会被注册进来
    decoder_map = build_decoder_routing_table(args.timeout)
    if not decoder_map:
        # 如果没有任何可用解码器，程序没有继续运行的意义，直接退出
        LOGGER.error("系统中未检测到任何可用的解码器底层工具，请先安装 (例如: libjxl-tools, libavif-bin 等)")
        return 2

    # 初始化一个空列表，用于收集需要解码的文件路径
    tasks: list[Path] = []
    
    # 如果用户输入的是一个目录，则需要遍历扫描该目录
    if args.input.is_dir():
        # 使用 rglob("*") 递归扫描目录下所有文件和子目录，并筛选出是文件且扩展名在支持列表里的
        for path in args.input.rglob("*"):
            if path.is_file() and path.suffix.lower() in decoder_map:
                tasks.append(path)  # 将符合条件的文件路径加入任务列表
    else:
        # 如果用户输入的是单独的一个文件
        if args.input.suffix.lower() in decoder_map:
            tasks.append(args.input)  # 如果扩展名支持，加入任务列表
        else:
            # 扩展名不支持，记录错误，可能原因包括：真的不支持，或者底层工具没安装导致没进入 decoder_map
            LOGGER.error("不支持的文件格式或对应的解码器未安装: %s", args.input)
            return 2
            
    # 如果扫描后发现任务列表为空，给出提示并正常退出（退出码 0）
    if not tasks:
        LOGGER.info("未在输入路径中找到受支持且可用解码的压缩图像文件")
        return 0

    # 确保输出目录存在，如果不存在则自动级联创建（parents=True）
    args.output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("发现 %d 个待解码文件，准备开始多线程解码...", len(tasks))

    success_count = 0  # 初始化成功解码的计数器
    failure_count = 0  # 初始化解码失败的计数器

    # 使用上下文管理器创建 ThreadPoolExecutor 线程池对象。max_workers 决定了最大并发数
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # future_to_path 是一个字典，用于保存 Future 对象（代表未来完成的任务）与原始文件路径的映射，方便追踪
        future_to_path = {}
        
        # 遍历所有收集到的任务路径
        for input_path in tasks:
            # 通过后缀名查路由表，获取对应的解码器实例
            codec = decoder_map[input_path.suffix.lower()]
            # 构造重建图像的输出路径：放在 output_dir 下，名字保持一致，但后缀强制改为 .png
            recon_path = args.output_dir / f"{input_path.stem}.png"
            
            # 将具体的任务函数及参数提交给线程池。
            # submit 函数非阻塞，会立即返回一个 Future 对象。
            future = executor.submit(decode_single_file, input_path, recon_path, codec)
            # 将返回的 Future 对象作为 key，原始路径作为 value 存入映射表
            future_to_path[future] = input_path

        # 使用 tqdm 包装 as_completed(future_to_path)，实现在控制台显示进度条
        # as_completed 会返回一个迭代器，每当有任何一个线程完成任务，它就会产出那个已完成的 Future
        for future in tqdm(as_completed(future_to_path), total=len(tasks), desc="解码进度"):
            # 获取该 Future 的执行结果（即 decode_single_file 函数的返回值）
            _path, success, _msg = future.result()
            
            if success:
                success_count += 1  # 若成功，成功计数 +1
            else:
                failure_count += 1  # 若失败，失败计数 +1

    # 在所有线程执行完毕后，进行汇总报告
    LOGGER.info("解码任务全部结束。成功: %d, 失败: %d", success_count, failure_count)
    
    # 只要存在失败的解码任务，就返回非零状态码 1（表示内部逻辑出错或部分失败）；全部成功则返回 0
    return 1 if failure_count > 0 else 0


if __name__ == "__main__":
    # Python 程序的标准入口保护：只有当脚本被直接执行时，才会调用 main() 函数，
    # 而如果作为模块被 import 则不会执行。将 main() 的返回值传递给 sys.exit() 以正确向操作系统反馈退出状态。
    sys.exit(main())