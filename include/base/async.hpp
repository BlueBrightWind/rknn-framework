#ifndef ASYNC_MODULE
#define ASYNC_MODULE

#include <base/model.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>

using namespace std;

/*
 *! @brief Encapsulation of multithreaded asynchronous inference.
 *  @param process() Return the output results based on the given input data.
 *  @attention The internal buffer is shallow-copied. If using pointers or structures containing pointers,
 *  be mindful of when the memory pointed to by the pointers is released. Make sure to release the memory
 *  after executing process().
 */
template <class input_type, class output_type>
class AsyncModule {
   private:
    bool stop_threads;
    size_t thread_num;
    vector<thread> threads;

    size_t buffer_length;

    mutex input_mtx;
    atomic<size_t> input_index;
    condition_variable input_producer;
    condition_variable input_consumer;
    queue<pair<size_t, input_type>> input_buffer;

    mutex output_mtx;
    atomic<size_t> output_index;
    condition_variable output_producer;
    condition_variable output_consumer;
    unordered_map<size_t, output_type> output_buffer;

   private:
    void worker(int worker_id);

   protected:
    virtual output_type process(input_type data, int worker_id = 0) = 0;

   public:
    AsyncModule();
    ~AsyncModule();
    void init(size_t buffer_length, size_t thread_num);
    void destroy();
    void put(input_type data);
    output_type get();
};

template <class input_type, class output_type>
AsyncModule<input_type, output_type>::AsyncModule() {
    this->buffer_length = 0;
    this->thread_num = 0;
    this->input_index.store(0);
    this->output_index.store(0);
    this->stop_threads = true;
}

template <class input_type, class output_type>
AsyncModule<input_type, output_type>::~AsyncModule() {
    this->destroy();
}

template <class input_type, class output_type>
void AsyncModule<input_type, output_type>::init(size_t buffer_length, size_t thread_num) {
    this->destroy();
    this->buffer_length = buffer_length;
    this->thread_num = thread_num;
    this->stop_threads = false;
    this->input_index.store(0);
    this->output_index.store(0);
    this->threads.clear();
    for (int i = 0; i < this->thread_num; i++) {
        threads.emplace_back(thread(&AsyncModule::worker, this, i));
    }
}

template <class input_type, class output_type>
void AsyncModule<input_type, output_type>::destroy() {
    this->stop_threads = true;
    input_consumer.notify_all();
    output_producer.notify_all();
    for (int i = 0; i < thread_num; i++)
        threads[i].join();
    this->buffer_length = 0;
    this->thread_num = 0;
    this->input_index.store(0);
    this->output_index.store(0);
}

template <class input_type, class output_type>
void AsyncModule<input_type, output_type>::put(input_type data) {
    size_t seq = input_index.fetch_add(1);
    unique_lock<mutex> lock(input_mtx);
    input_producer.wait(lock, [&]() {
        return input_buffer.size() < buffer_length;
    });
    input_buffer.push({seq, data});
    input_consumer.notify_one();
}

template <class input_type, class output_type>
void AsyncModule<input_type, output_type>::worker(int worker_id) {
    while (!this->stop_threads) {
        unique_lock<mutex> i_lock(input_mtx);
        input_consumer.wait(i_lock, [&]() {
            if (stop_threads)
                return true;
            return !input_buffer.empty();
        });
        if (stop_threads)
            return;
        size_t seq = input_buffer.front().first;
        input_type data = input_buffer.front().second;
        input_buffer.pop();
        input_producer.notify_one();
        i_lock.unlock();
        const output_type output = process(data, worker_id);
        unique_lock<mutex> o_lock(output_mtx);
        output_producer.wait(o_lock, [&]() {
            size_t last_seq = output_index.load();
            if (stop_threads)
                return true;
            if (seq < last_seq)
                return true;
            return (seq - last_seq) < buffer_length;
        });
        if (stop_threads)
            return;
        output_buffer[seq] = output;
        output_consumer.notify_one();
        o_lock.unlock();
    }
    return;
}

template <class input_type, class output_type>
output_type AsyncModule<input_type, output_type>::get() {
    size_t seq = output_index.fetch_add(1);
    unique_lock<mutex> lock(output_mtx);
    output_consumer.wait(lock, [&]() {
        return output_buffer.find(seq) != output_buffer.end();
    });
    output_type result = output_buffer[seq];
    output_buffer.erase(seq);
    output_producer.notify_one();
    return result;
}

#endif  // ASYNC_BASE_MODEL
