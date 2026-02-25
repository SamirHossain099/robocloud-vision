import runpod


def handler(event):
    return {"status": "hello from vision"}


runpod.serverless.start({"handler": handler})
