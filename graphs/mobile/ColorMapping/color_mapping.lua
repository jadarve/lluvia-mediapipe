local builder = ll.class(ll.ContainerNodeBuilder)

builder.name = 'mediapipe/examples/ColorMapping'
builder.doc = [[
Color mapping example.
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

    -- ColorMap
    local ColorMap = ll.createContainerNode('lluvia/viz/colormap/ColorMap')
    ColorMap:bind('in_image', out_gray)
    ColorMap:setParameter('colormap', 'inferno')
    ColorMap:setParameter('min_value', 0)
    ColorMap:setParameter('max_value', 255)
    ColorMap:setParameter('alpha', 0.0)
    ColorMap:setParameter('reverse', 0)
    ColorMap:init()
    node:bindNode('ColorMap', ColorMap)

    local out_rgba = ColorMap:getPort('out_rgba')

    -- bind the output
    node:bind('out_image', out_rgba)

    ll.logd(node.descriptor.builderName, 'onNodeInit: finish')

end


function builder.onNodeRecord(node, cmdBuffer)

    ll.logd(node.descriptor.builderName, 'onNodeRecord')

    local BGRA2Gray = node:getNode('BGRA2Gray')
    local ColorMap = node:getNode('ColorMap')

    BGRA2Gray:record(cmdBuffer)
    cmdBuffer:memoryBarrier()

    ColorMap:record(cmdBuffer)
    cmdBuffer:memoryBarrier()

    ll.logd(node.descriptor.builderName, 'onNodeRecord: finish')
end


ll.registerNodeBuilder(builder)
