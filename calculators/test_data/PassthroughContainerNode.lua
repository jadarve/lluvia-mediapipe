local builder = ll.class(ll.ContainerNodeBuilder)

builder.name = 'mediapipe/test/PassthroughContainerNode'
builder.doc = [[
A passthrough calculator copying the content of one image to the output.

]]

function builder.newDescriptor()

    local desc = ll.ContainerNodeDescriptor.new()

    desc.builderName = builder.name

    -- TODO: Parameters

    return desc
end


function builder.onNodeInit(node)

    ll.logd(node.descriptor.builderName, 'onNodeInit')

    local inputCounter = 0
    while true do

        local inputName = string.format('in_image_%d', inputCounter)
        if not node:hasPort(inputName) then
            break
        end
        
        local in_image = node:getPort(inputName)

        local memory = in_image.memory

        -- creates a copy of in_image
        local out_image = memory:createImageView(in_image.imageDescriptor, in_image.descriptor)
        out_image:changeImageLayout(ll.ImageLayout.General)

        -- bind the outuput
        node:bind(string.format('out_image_%d', inputCounter), out_image)

        inputCounter = inputCounter + 1

    end

    ll.logd(node.descriptor.builderName, 'onNodeInit: finish')

end


function builder.onNodeRecord(node, cmdBuffer)

    ll.logd(node.descriptor.builderName, 'onNodeRecord')

    local inputCounter = 0
    while true do
        
        local inputName = string.format('in_image_%d', inputCounter)
        if not node:hasPort(inputName) then
            break
        end
        
        local in_image = node:getPort(inputName)
        local out_image = node:getPort(string.format('out_image_%d', inputCounter))

        cmdBuffer:changeImageLayout(in_image.image, ll.ImageLayout.TransferSrcOptimal)
        cmdBuffer:changeImageLayout(out_image.image, ll.ImageLayout.TransferDstOptimal)
        cmdBuffer:memoryBarrier()
        cmdBuffer:copyImageToImage(in_image.image, out_image.image)
        cmdBuffer:memoryBarrier()
        cmdBuffer:changeImageLayout(in_image.image, ll.ImageLayout.General)
        cmdBuffer:changeImageLayout(out_image.image, ll.ImageLayout.General)

        inputCounter = inputCounter + 1
    end

    ll.logd(node.descriptor.builderName, 'onNodeRecord: finish')
end


ll.registerNodeBuilder(builder)
