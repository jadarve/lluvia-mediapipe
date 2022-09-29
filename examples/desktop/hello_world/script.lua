local builder = ll.class(ll.ContainerNodeBuilder)

builder.name = 'mediapipe/examples/HelloWorldContainerNode'
builder.doc = [[
A passthrough calculator copying the content of one image to the output.

]]

function builder.newDescriptor()

    local desc = ll.ContainerNodeDescriptor.new()

    desc.builderName = builder.name

    -- No need to declare the input
    local in_image = ll.PortDescriptor.new(0, 'in_image', ll.PortDirection.In, ll.PortType.ImageView)
    desc:addPort(in_image)

    local out_image = ll.PortDescriptor.new(0, 'out_image', ll.PortDirection.Out, ll.PortType.ImageView)
    desc:addPort(out_image)

    -- TODO: Parameters
    -- desc:setParameter('levels', 1)

    return desc
end


function builder.onNodeInit(node)

    ll.logd(node.descriptor.builderName, 'onNodeInit')

    local in_image = node:getPort('in_image')

    local memory = in_image.memory

    -- creates a copy of in_image
    local out_image = memory:createImageView(in_image.imageDescriptor, in_image.descriptor)
    out_image:changeImageLayout(ll.ImageLayout.General)
    -- bind the outuput
    node:bind('out_image', out_image)

    ll.logd(node.descriptor.builderName, 'onNodeInit: finish')

end


function builder.onNodeRecord(node, cmdBuffer)

    ll.logd(node.descriptor.builderName, 'onNodeRecord')

    local in_image = node:getPort('in_image')
    local out_image = node:getPort('out_image')

    cmdBuffer:changeImageLayout(in_image.image, ll.ImageLayout.TransferSrcOptimal)
    cmdBuffer:changeImageLayout(out_image.image, ll.ImageLayout.TransferDstOptimal)
    cmdBuffer:memoryBarrier()
    cmdBuffer:copyImageToImage(in_image.image, out_image.image)
    cmdBuffer:memoryBarrier()
    cmdBuffer:changeImageLayout(in_image.image, ll.ImageLayout.General)
    cmdBuffer:changeImageLayout(out_image.image, ll.ImageLayout.General)

    ll.logd(node.descriptor.builderName, 'onNodeRecord: finish')
end


ll.registerNodeBuilder(builder)
