local builder = ll.class(ll.ContainerNodeBuilder)

builder.name = 'mediapipe/examples/FlowFilter'
builder.doc = [[
A calculator transforming computing optical flow using the Horn and Schunck algorithm.

]]

function builder.newDescriptor()

    local desc = ll.ContainerNodeDescriptor.new()

    desc.builderName = builder.name

    -- No need to declare the input
    local in_image = ll.PortDescriptor.new(0, 'in_image', ll.PortDirection.In, ll.PortType.ImageView)
    desc:addPort(in_image)

    local out_image = ll.PortDescriptor.new(1, 'out_image', ll.PortDirection.Out, ll.PortType.ImageView)
    desc:addPort(out_image)

    return desc
end


function builder.onNodeInit(node)

    ll.logd(node.descriptor.builderName, 'onNodeInit')

    local in_image = node:getPort('in_image')

    -- BGRA to Gray
    local BGRA2Gray = ll.createComputeNode('lluvia/color/BGRA2Gray')
    BGRA2Gray:bind('in_bgra', in_image)
    BGRA2Gray:init()
    node:bindNode('BGRA2Gray', BGRA2Gray)

    local out_gray = BGRA2Gray:getPort('out_gray')

    -- FlowFilter
    local FlowFilter = ll.createContainerNode('lluvia/opticalflow/flowfilter/FlowFilter')
    FlowFilter:setParameter('gamma', 0.001)
    FlowFilter:setParameter('gamma_low', 0.0001)
    FlowFilter:setParameter('levels', 1)
    FlowFilter:setParameter('max_flow', 4)
    FlowFilter:setParameter('smooth_iterations', 1)
    FlowFilter:setParameter('float_precision', ll.FloatPrecision.FP16)
    FlowFilter:bind('in_gray', out_gray)

    FlowFilter:init()
    node:bindNode('FlowFilter', FlowFilter)
    
    local out_flow = FlowFilter:getPort('out_flow')

    -- Flow to RGBA
    local Flow2RGBA = ll.createComputeNode('lluvia/viz/Flow2RGBA')
    Flow2RGBA:setParameter('max_flow', 16.0)
    Flow2RGBA:bind('in_flow', out_flow)
    Flow2RGBA:init()
    node:bindNode('Flow2RGBA', Flow2RGBA)

    local flowRGBA = Flow2RGBA:getPort('out_rgba')

    -- RGBA to BGRA
    local RGBA2BGRA = ll.createComputeNode('lluvia/color/RGBA2BGRA')
    RGBA2BGRA:bind('in_rgba', flowRGBA)
    RGBA2BGRA:init()
    node:bindNode('RGBA2BGRA', RGBA2BGRA)

    local out_bgra = RGBA2BGRA:getPort('out_bgra')
    
    -- bind the output
    node:bind('out_image', out_bgra)

    ll.logd(node.descriptor.builderName, 'onNodeInit: finish')

end


function builder.onNodeRecord(node, cmdBuffer)

    ll.logd(node.descriptor.builderName, 'onNodeRecord')

    local BGRA2Gray = node:getNode('BGRA2Gray')
    local FlowFilter = node:getNode('FlowFilter')
    local Flow2RGBA = node:getNode('Flow2RGBA')
    local RGBA2BGRA = node:getNode('RGBA2BGRA')

    BGRA2Gray:record(cmdBuffer)
    cmdBuffer:memoryBarrier()

    FlowFilter:record(cmdBuffer)
    -- no memory barrier here as HS container node puts a barrier at the end

    Flow2RGBA:record(cmdBuffer)
    cmdBuffer:memoryBarrier()

    RGBA2BGRA:record(cmdBuffer)
    cmdBuffer:memoryBarrier()

    ll.logd(node.descriptor.builderName, 'onNodeRecord: finish')
end


ll.registerNodeBuilder(builder)
