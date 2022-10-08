local builder = ll.class(ll.ContainerNodeBuilder)

builder.name = 'lluvia/mediapipe/LluviaCalculator'
builder.doc = [[
The lluvia calculator

]]

function builder.newDescriptor()

    local desc = ll.ContainerNodeDescriptor.new()

    desc.builderName = builder.name

    -- No need to declare the input
    local in_image = ll.PortDescriptor.new(0, 'in_image', ll.PortDirection.In, ll.PortType.ImageView)
    desc:addPort(in_image)

    -- TODO: Parameters
    -- desc:setParameter('levels', 1)

    return desc
end


function builder.onNodeInit(node)

    ll.logd(node.descriptor.builderName, 'onNodeInit')

    local in_image = node:getPort('in_image')

    local RGBA2Gray = ll.createComputeNode('lluvia/color/RGBA2Gray')
    RGBA2Gray:bind('in_rgba', in_image)
    RGBA2Gray:init()

    node:bindNode('RGBA2Gray', RGBA2Gray)

    -- bind the outuput
    node:bind('out_image', RGBA2Gray:getPort('out_gray'))

    ll.logd(node.descriptor.builderName, 'onNodeInit: finish')

end


function builder.onNodeRecord(node, cmdBuffer)

    ll.logd(node.descriptor.builderName, 'onNodeRecord')

    local RGBA2Gray = node:getNode('RGBA2Gray')
    RGBA2Gray:record(cmdBuffer)

    ll.logd(node.descriptor.builderName, 'onNodeRecord: finish')
end


ll.registerNodeBuilder(builder)
