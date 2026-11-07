// Local OCR cross-check. Both readings remain fallible, untrusted image data.
// Normalize oversized glyphs before recognition; keep original alternatives rather than
// treating confidence, agreement, or a preferred rendering as ground truth.
import Foundation
import ImageIO
import Vision
import CoreGraphics

struct Candidate: Encodable { let text: String; let confidence: Float }
struct Line: Encodable { let candidates: [Candidate] }
struct Output: Encodable { let lines: [Line] }

func recognize(_ image: CGImage) throws -> [VNRecognizedTextObservation] {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.recognitionLanguages = ["ko-KR", "en-US"]
    request.usesLanguageCorrection = false
    try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
    return request.results ?? []
}

func normalized(_ image: CGImage, observations: [VNRecognizedTextObservation]) -> CGImage? {
    let heights = observations.map { $0.boundingBox.height * Double(image.height) }.sorted()
    guard !heights.isEmpty else { return nil }
    let height = heights[heights.count / 2]
    guard height > 32 else { return nil }
    let scale = 22 / height
    let w = max(1, Int((Double(image.width) * scale).rounded()))
    let h = max(1, Int((Double(image.height) * scale).rounded()))
    guard let context = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
    context.interpolationQuality = .high
    context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
    return context.makeImage()
}

guard CommandLine.arguments.count == 2,
      let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: CommandLine.arguments[1]) as CFURL, nil),
      let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
      let width = properties[kCGImagePropertyPixelWidth] as? Int,
      let height = properties[kCGImagePropertyPixelHeight] as? Int,
      width > 0, height > 0, width <= 16384, height <= 16384,
      width <= 20_000_000 / height,
      let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { exit(2) }

do {
    let original = try recognize(image)
    var observations = original
    if let smaller = normalized(image, observations: original) {
        do { observations = try recognize(smaller) + original }
        catch { fputs("normalized OCR failed; preserving original reading\n", stderr) }
    }
    let lines = observations.map { observation -> Line in
        let candidates = observation.topCandidates(3).map { Candidate(text: $0.string, confidence: $0.confidence) }
        return Line(candidates: candidates)
    }
    let data = try JSONEncoder().encode(Output(lines: lines))
    guard data.count <= 64 * 1024 else { exit(2) }
    print(String(decoding: data, as: UTF8.self))
} catch {
    fputs("OCR recognition failed\n", stderr)
    exit(2)
}
