local builder = ll.class(ll.ContainerNodeBuilder)

builder.name = 'mediapipe/examples/BGRA2Gray'
builder.doc = [[
A calculator transforming an RGBA image to grayscale.

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

    local BGRA2Gray = ll.createComputeNode('lluvia/color/BGRA2Gray')
    BGRA2Gray:bind('in_bgra', in_image)
    BGRA2Gray:init()

    -- bind the node
    node:bindNode('BGRA2Gray', BGRA2Gray)

    local out_image = BGRA2Gray:getPort('out_gray')
    
    -- bind the output
    node:bind('out_image', out_image)

    ll.logd(node.descriptor.builderName, 'onNodeInit: finish')

end


function builder.onNodeRecord(node, cmdBuffer)

    ll.logd(node.descriptor.builderName, 'onNodeRecord')

    local BGRA2Gray = node:getNode('BGRA2Gray')
    BGRA2Gray:record(cmdBuffer)

    ll.logd(node.descriptor.builderName, 'onNodeRecord: finish')
end


ll.registerNodeBuilder(builder)
