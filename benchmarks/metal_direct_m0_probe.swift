import Foundation
import Metal

struct Args {
    var values: [String: String] = [:]
}

func parseArgs() -> Args {
    var parsed = Args()
    let argv = Array(CommandLine.arguments.dropFirst())
    var index = 0
    while index < argv.count {
        let key = argv[index]
        if key.hasPrefix("--"), index + 1 < argv.count {
            parsed.values[String(key.dropFirst(2))] = argv[index + 1]
            index += 2
        } else {
            index += 1
        }
    }
    return parsed
}

func required(_ args: Args, _ key: String) throws -> String {
    if let value = args.values[key] {
        return value
    }
    throw NSError(domain: "metal_direct_m0_probe", code: 1, userInfo: [NSLocalizedDescriptionKey: "Missing argument \(key)"])
}

func readFloatArray(path: String) throws -> [Float] {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    return data.withUnsafeBytes { rawBuffer in
        Array(rawBuffer.bindMemory(to: Float.self))
    }
}

func writeFloatArray(path: String, values: [Float]) throws {
    let data = values.withUnsafeBufferPointer { buffer in
        Data(buffer: buffer)
    }
    try data.write(to: URL(fileURLWithPath: path))
}

let args = parseArgs()

do {
    let metalSource = try String(contentsOfFile: required(args, "metal-source"), encoding: .utf8)
    let kernelKind = try required(args, "kernel")
    let queries = try readFloatArray(path: required(args, "queries"))
    let queryGroupSums = try readFloatArray(path: required(args, "query-group-sums"))
    let fused: [Float] = {
        guard let path = args.values["fused"] else { return [] }
        return (try? readFloatArray(path: path)) ?? []
    }()
    let bias = try readFloatArray(path: required(args, "bias"))
    let payloadWords: [UInt32] = {
        guard let path = args.values["payload"] else { return [] }
        let data = try! Data(contentsOf: URL(fileURLWithPath: path))
        return data.withUnsafeBytes { rawBuffer in
            Array(rawBuffer.bindMemory(to: UInt32.self))
        }
    }()
    let scales: [Float] = {
        guard let path = args.values["scales"] else { return [] }
        let data = try! Data(contentsOf: URL(fileURLWithPath: path))
        return data.withUnsafeBytes { rawBuffer in
            Array(rawBuffer.bindMemory(to: Float.self))
        }
    }()
    let outputPath = try required(args, "output")
    let batchCount = Int(try required(args, "batch-count"))!
    let queryCount = Int(try required(args, "query-count"))!
    let paddedHeadDim = Int(try required(args, "padded-head-dim"))!
    let tokenCount = Int(try required(args, "token-count"))!
    let numGroups = Int(try required(args, "num-groups"))!
    let wordsPerGroup = Int(args.values["words-per-group"] ?? "0")!
    let queryScale = Float(try required(args, "query-scale"))!
    let warmupIters = Int(try required(args, "warmup-iters"))!
    let benchIters = Int(try required(args, "bench-iters"))!

    guard let device = MTLCreateSystemDefaultDevice() else {
        throw NSError(domain: "metal_direct_m0_probe", code: 2, userInfo: [NSLocalizedDescriptionKey: "No Metal device available"])
    }
    let library = try device.makeLibrary(source: metalSource, options: nil)
    let functionName: String
    switch kernelKind {
    case "transposed":
        functionName = "direct_m0_logits_transposed_affine"
    case "transposed_tiled":
        functionName = "direct_m0_logits_transposed_tiled_affine"
    case "packed_group_major_8bit":
        functionName = "direct_m0_logits_packed_group_major_affine_8bit"
    default:
        functionName = "direct_m0_logits_flat_affine"
    }
    guard let function = library.makeFunction(name: functionName) else {
        throw NSError(domain: "metal_direct_m0_probe", code: 3, userInfo: [NSLocalizedDescriptionKey: "Missing function \(functionName)"])
    }
    let pipeline = try device.makeComputePipelineState(function: function)
    guard let commandQueue = device.makeCommandQueue() else {
        throw NSError(domain: "metal_direct_m0_probe", code: 4, userInfo: [NSLocalizedDescriptionKey: "Unable to create command queue"])
    }

    let outputCount = batchCount * queryCount * tokenCount
    let queriesBuffer = device.makeBuffer(bytes: queries, length: queries.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
    let queryGroupSumsBuffer = device.makeBuffer(bytes: queryGroupSums, length: queryGroupSums.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
    let fusedBuffer = fused.isEmpty ? nil : device.makeBuffer(bytes: fused, length: fused.count * MemoryLayout<Float>.stride, options: .storageModeShared)
    let payloadBuffer = payloadWords.isEmpty ? nil : device.makeBuffer(bytes: payloadWords, length: payloadWords.count * MemoryLayout<UInt32>.stride, options: .storageModeShared)
    let scalesBuffer = scales.isEmpty ? nil : device.makeBuffer(bytes: scales, length: scales.count * MemoryLayout<Float>.stride, options: .storageModeShared)
    let biasBuffer = device.makeBuffer(bytes: bias, length: bias.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
    let outputBuffer = device.makeBuffer(length: outputCount * MemoryLayout<Float>.stride, options: .storageModeShared)!

    var paddedHeadDimU32 = UInt32(paddedHeadDim)
    var tokenCountU32 = UInt32(tokenCount)
    var queryCountU32 = UInt32(queryCount)
    var numGroupsU32 = UInt32(numGroups)
    var wordsPerGroupU32 = UInt32(wordsPerGroup)
    var queryScaleF32 = queryScale
    let paddedHeadDimBuffer = device.makeBuffer(bytes: &paddedHeadDimU32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    let tokenCountBuffer = device.makeBuffer(bytes: &tokenCountU32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    let queryCountBuffer = device.makeBuffer(bytes: &queryCountU32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    let numGroupsBuffer = device.makeBuffer(bytes: &numGroupsU32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    let wordsPerGroupBuffer = device.makeBuffer(bytes: &wordsPerGroupU32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    let queryScaleBuffer = device.makeBuffer(bytes: &queryScaleF32, length: MemoryLayout<Float>.stride, options: .storageModeShared)!

    let threadsPerGroupWidth = min(max(1, pipeline.threadExecutionWidth), tokenCount)
    let maxThreads = max(1, pipeline.maxTotalThreadsPerThreadgroup)
    let threadsPerGroupHeight = max(1, min(queryCount, maxThreads / threadsPerGroupWidth))
    let threadsPerThreadgroup = MTLSize(width: threadsPerGroupWidth, height: threadsPerGroupHeight, depth: 1)
    let threadsPerGrid = MTLSize(width: tokenCount, height: queryCount, depth: 1)

    func encodePass() throws -> MTLCommandBuffer {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw NSError(domain: "metal_direct_m0_probe", code: 5, userInfo: [NSLocalizedDescriptionKey: "Unable to create command buffer"])
        }
        encoder.setComputePipelineState(pipeline)
        for batchIndex in 0..<batchCount {
            let queryOffset = batchIndex * queryCount * paddedHeadDim * MemoryLayout<Float>.stride
            let queryGroupSumsOffset = batchIndex * queryCount * numGroups * MemoryLayout<Float>.stride
            let outputOffset = batchIndex * queryCount * tokenCount * MemoryLayout<Float>.stride
            let fusedOffset: Int
            if kernelKind == "transposed" || kernelKind == "transposed_tiled" {
                fusedOffset = batchIndex * paddedHeadDim * tokenCount * MemoryLayout<Float>.stride
            } else {
                fusedOffset = batchIndex * tokenCount * paddedHeadDim * MemoryLayout<Float>.stride
            }
            let biasOffset = batchIndex * numGroups * tokenCount * MemoryLayout<Float>.stride
            encoder.setBuffer(queriesBuffer, offset: queryOffset, index: 0)
            encoder.setBuffer(queryGroupSumsBuffer, offset: queryGroupSumsOffset, index: 1)
            if kernelKind == "packed_group_major_8bit" {
                let payloadOffset = batchIndex * numGroups * tokenCount * wordsPerGroup * MemoryLayout<UInt32>.stride
                let scalesOffset = batchIndex * tokenCount * numGroups * MemoryLayout<Float>.stride
                encoder.setBuffer(payloadBuffer, offset: payloadOffset, index: 2)
                encoder.setBuffer(scalesBuffer, offset: scalesOffset, index: 3)
                encoder.setBuffer(biasBuffer, offset: biasOffset, index: 4)
                encoder.setBuffer(outputBuffer, offset: outputOffset, index: 5)
                encoder.setBuffer(tokenCountBuffer, offset: 0, index: 6)
                encoder.setBuffer(queryCountBuffer, offset: 0, index: 7)
                encoder.setBuffer(numGroupsBuffer, offset: 0, index: 8)
                encoder.setBuffer(wordsPerGroupBuffer, offset: 0, index: 9)
                encoder.setBuffer(queryScaleBuffer, offset: 0, index: 10)
            } else {
                encoder.setBuffer(fusedBuffer, offset: fusedOffset, index: 2)
                encoder.setBuffer(biasBuffer, offset: biasOffset, index: 3)
                encoder.setBuffer(outputBuffer, offset: outputOffset, index: 4)
                encoder.setBuffer(paddedHeadDimBuffer, offset: 0, index: 5)
                encoder.setBuffer(tokenCountBuffer, offset: 0, index: 6)
                encoder.setBuffer(queryCountBuffer, offset: 0, index: 7)
                encoder.setBuffer(numGroupsBuffer, offset: 0, index: 8)
                encoder.setBuffer(queryScaleBuffer, offset: 0, index: 9)
            }
            encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        }
        encoder.endEncoding()
        return commandBuffer
    }

    for _ in 0..<max(0, warmupIters) {
        let commandBuffer = try encodePass()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    let start = DispatchTime.now().uptimeNanoseconds
    for _ in 0..<max(1, benchIters) {
        let commandBuffer = try encodePass()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    let end = DispatchTime.now().uptimeNanoseconds
    let elapsedMs = Double(end - start) / 1_000_000.0 / Double(max(1, benchIters))

    let outputPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: outputCount)
    let output = Array(UnsafeBufferPointer(start: outputPointer, count: outputCount))
    try writeFloatArray(path: outputPath, values: output)

    let payload: [String: Any] = [
        "enabled": true,
        "kernel": kernelKind,
        "avg_ms": elapsedMs,
    ]
    let data = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write("\n".data(using: .utf8)!)
} catch {
    FileHandle.standardError.write("\(error)\n".data(using: .utf8)!)
    exit(1)
}
